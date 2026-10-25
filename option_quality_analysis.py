import json
import os
import re
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def assess_option_quality(option_zh, option_mt):
    prompt = f"""You are a translation quality assessor.
Compare the original Chinese medical option with its machine translation.

Original Chinese: {option_zh}
Machine translation: {option_mt}

Rate the translation quality:
1 = Good (meaning preserved, terminology correct)
2 = Partial (meaning partially preserved, some errors)
3 = Bad (meaning lost or completely wrong)

Reply with ONLY a JSON:
{{"score": 1, "reason": "one sentence"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0
    )
    text = response.choices[0].message.content.strip()
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {"score": 2, "reason": "parse error"}

# Load data
with open("gold_set_200_translated.json") as f:
    questions = json.load(f)
with open("accuracy_results_200.json") as f:
    acc_data = json.load(f)

acc_by_id = {r["id"]: r for r in acc_data}

print("Assessing option translation quality for 200 questions...")
results = []

for i, q in enumerate(questions):
    print(f"Q{i+1}/200...", end=" ", flush=True)

    option_scores = []
    for key in q["option"]:
        zh_val = q["option"][key]
        mt_val = q["option_mt"].get(key, "")
        if zh_val.strip() and mt_val.strip():
            assessment = assess_option_quality(zh_val, mt_val)
            option_scores.append(assessment["score"])

    avg_option_score = np.mean(option_scores) if option_scores else 2
    worst_option_score = max(option_scores) if option_scores else 2

    acc = acc_by_id.get(q["id"], {})
    c2_correct = int(acc.get("pred_c2", "") == q["answer"])
    c3_correct = int(acc.get("pred_c3", "") == q["answer"])
    c5_correct = int(acc.get("pred_c5", "") == q["answer"])

    results.append({
        "id": q["id"],
        "avg_option_score": round(avg_option_score, 2),
        "worst_option_score": worst_option_score,
        "c2_correct": c2_correct,
        "c3_correct": c3_correct,
        "c5_correct": c5_correct
    })

    print(f"avg={avg_option_score:.1f}")

with open("option_quality_analysis_200.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Analysis
print("\n=== Option Quality vs Accuracy (200 questions) ===")
good = [r for r in results if r["avg_option_score"] <= 1.5]
partial = [r for r in results if 1.5 < r["avg_option_score"] <= 2.5]
bad = [r for r in results if r["avg_option_score"] > 2.5]

for group, label in [(good, "Good options (score≤1.5)"),
                     (partial, "Partial options (1.5<score≤2.5)"),
                     (bad, "Bad options (score>2.5)")]:
    if not group:
        continue
    c2 = np.mean([r["c2_correct"] for r in group])
    c3 = np.mean([r["c3_correct"] for r in group])
    c5 = np.mean([r["c5_correct"] for r in group])
    print(f"\n{label} (n={len(group)}):")
    print(f"  C2: {c2:.3f} | C3: {c3:.3f} | C5: {c5:.3f}")

# Correlation
from scipy import stats
option_scores = [r["avg_option_score"] for r in results]
c2_accs = [r["c2_correct"] for r in results]
r, p = stats.pearsonr(option_scores, c2_accs)
print(f"\n=== Correlation: Option Quality vs C2 Accuracy ===")
print(f"Pearson r = {r:.3f}, p = {p:.4f}")