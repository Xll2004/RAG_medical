import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from collections import Counter

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load accuracy results to find C1✓ C2✗ cases
with open("results/accuracy_results_200.json") as f:
    acc_data = json.load(f)

target_ids = [r["id"] for r in acc_data
              if r["pred_c1"] == r["correct_answer"]
              and r["pred_c2"] != r["correct_answer"]]

print(f"Questions with C1 correct but C2 wrong: {len(target_ids)}")

# Load all data
with open("data/gold_set_200_translated.json") as f:
    questions = json.load(f)
with open("results/results_condition1_200.json") as f:
    c1 = json.load(f)
with open("results/results_condition2_200.json") as f:
    c2 = json.load(f)
with open("results/error_annotations_200.json") as f:
    errors = json.load(f)

# Build lookup by ID
q_by_id = {q["id"]: q for q in questions}
c1_by_id = {r["id"]: r for r in c1}
c2_by_id = {r["id"]: r for r in c2}
err_by_id = {e["id"]: e for e in errors}

def analyze_failure(qid):
    q = q_by_id[qid]
    err = err_by_id.get(qid, {})

    prompt = f"""You are a translation error analyst. A Chinese medical question was correctly answered when given a high-quality English translation (C1), but incorrectly answered when given a machine translation (C2).

Chinese question: {q['question']}
C1 (high quality): {c1_by_id[qid]['question_en_baseline']}
C2 (machine translation): {q['question_mt']}

C1 options: {json.dumps(q['option'], ensure_ascii=False)}
C2 options: {json.dumps(q['option_mt'], ensure_ascii=False)}

Known error types present: {err.get('error_types', [])}

What is the PRIMARY reason C2 failed? Choose ONE:
T1 = Terminology error was the main cause of retrieval/answer failure
T2 = Syntactic distortion was the main cause
T3 = Semantic drift was the main cause
MIX = Multiple errors equally contributed

Reply with ONLY a JSON:
{{"primary_cause": "T1/T2/T3/MIX", "explanation": "one sentence reason"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )

    text = response.choices[0].message.content.strip()
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {"primary_cause": "MIX", "explanation": "parse error"}

print(f"Analyzing {len(target_ids)} failure cases...")
results = []

for qid in target_ids:
    print(f"Analyzing ID {qid}...", end=" ", flush=True)
    result = analyze_failure(qid)
    results.append({
        "id": qid,
        "question_zh": q_by_id[qid]["question"],
        "question_mt": q_by_id[qid]["question_mt"],
        "primary_cause": result["primary_cause"],
        "explanation": result["explanation"]
    })
    print(f"{result['primary_cause']} - {result['explanation'][:60]}")

# Save
with open("results/failure_analysis_200.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Summary
n = len(target_ids)
causes = Counter(r["primary_cause"] for r in results)
print(f"\n=== Primary Failure Causes (C1 correct, C2 wrong, n={n}) ===")
for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
    print(f"\n{cause}: {count}/{n} ({count/n:.1%})")
    for r in results:
        if r["primary_cause"] == cause:
            print(f"  ID {r['id']}: {r['explanation']}")