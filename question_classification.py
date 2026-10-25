import json
import os
import re
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_question(question_zh, options_zh):
    options_text = "\n".join([f"{k}: {v}" for k, v in options_zh.items()])
    prompt = f"""You are a medical education expert. Classify this Chinese medical exam question into ONE category.

Question: {question_zh}
Options: {options_text}

Categories:
1. FACTUAL_RECALL - Requires recalling a specific medical fact, term, drug, or definition
2. CLINICAL_REASONING - Requires analyzing patient symptoms/signs to reach a diagnosis or treatment decision
3. PHARMACOLOGY - About drug mechanisms, dosages, interactions, or contraindications
4. PROCEDURE - About medical procedures, techniques, or protocols
5. REGULATION - About medical laws, regulations, or administrative rules

Reply with ONLY a JSON:
{{"category": "CATEGORY_NAME", "explanation": "one sentence reason"}}"""

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
    return {"category": "FACTUAL_RECALL", "explanation": "parse error"}

# Load data
with open("gold_set_200_translated.json") as f:
    questions = json.load(f)

with open("accuracy_results_200.json") as f:
    acc_data = json.load(f)

acc_by_id = {r["id"]: r for r in acc_data}

print("Classifying 200 questions...")
classifications = []

for i, q in enumerate(questions):
    print(f"Q{i+1}/200...", end=" ", flush=True)
    result = classify_question(q["question"], q["option"])

    acc = acc_by_id.get(q["id"], {})
    c1_correct = int(acc.get("pred_c1", "") == q["answer"])
    c2_correct = int(acc.get("pred_c2", "") == q["answer"])
    c5_correct = int(acc.get("pred_c5", "") == q["answer"])

    classifications.append({
        "id": q["id"],
        "question_zh": q["question"],
        "category": result["category"],
        "explanation": result["explanation"],
        "c1_correct": c1_correct,
        "c2_correct": c2_correct,
        "c5_correct": c5_correct,
        "c1_c2_gap": c1_correct - c2_correct
    })

    print(f"{result['category']}")

with open("question_classifications_200.json", "w", encoding="utf-8") as f:
    json.dump(classifications, f, ensure_ascii=False, indent=2)

# Analysis
print("\n=== Category Distribution (200 questions) ===")
cat_counter = Counter(c["category"] for c in classifications)
for cat, count in sorted(cat_counter.items(), key=lambda x: -x[1]):
    items = [c for c in classifications if c["category"] == cat]
    c1_acc = sum(c["c1_correct"] for c in items) / len(items)
    c2_acc = sum(c["c2_correct"] for c in items) / len(items)
    c5_acc = sum(c["c5_correct"] for c in items) / len(items)
    gap = c1_acc - c2_acc
    print(f"\n{cat} (n={count}):")
    print(f"  C1: {c1_acc:.3f} | C2: {c2_acc:.3f} | C5: {c5_acc:.3f} | Gap C1-C2: {gap:+.3f}")