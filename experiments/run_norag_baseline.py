import json
import os
import time
import numpy as np
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_answer_choice_no_context(question, options):
    """Same prompt structure as run_accuracy.py's get_answer_choice,
    but with the Context section removed entirely (no retrieval)."""
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    prompt = f"""You are a medical expert. Answer the question using your own medical knowledge.
Only reply with a single letter (A, B, C, D, or E).
Question: {question}
Options:
{options_text}
Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0
    )
    text = response.choices[0].message.content.strip()
    return text[0].upper() if text else "X"


# Load gold answers and C1's already-translated questions/options.
# Reusing C1's translations (rather than re-translating) holds translation
# quality constant, isolating "presence of retrieval" as the only variable.
with open("data/gold_set_200_translated.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

with open("results/results_condition1_200.json", "r", encoding="utf-8") as f:
    c1 = json.load(f)

n = min(len(questions), len(c1), 200)
misaligned = sum(1 for i in range(n) if questions[i]["id"] != c1[i]["id"])
print(f"Using {n} questions, gold vs C1 ID misalignments: {misaligned}")

N_RUNS = 3
all_run_results = []

for run in range(N_RUNS):
    print(f"\n--- Run {run + 1}/{N_RUNS} ---")
    correct = 0
    run_results = []
    for i, q in enumerate(questions[:n]):
        question_en = c1[i]["question_en_baseline"]
        options_en = c1[i]["options_en"]
        pred = get_answer_choice_no_context(question_en, options_en)
        ans = q["answer"]
        correct += int(pred == ans)
        run_results.append({
            "id": q["id"],
            "correct_answer": ans,
            "pred": pred
        })
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{n} | no-RAG acc so far: {correct / (i + 1):.3f}")
        time.sleep(0.2)
    all_run_results.append(run_results)
    print(f"Run {run + 1}: no-RAG acc = {correct / n:.3f}")

# Majority vote across 3 runs, matching run_accuracy.py's protocol
final_results = []
for i in range(n):
    votes = [all_run_results[r][i]["pred"] for r in range(N_RUNS)]
    final_results.append({
        "id": all_run_results[0][i]["id"],
        "correct_answer": all_run_results[0][i]["correct_answer"],
        "pred_norag": Counter(votes).most_common(1)[0][0]
    })

with open("results/results_norag_baseline_200.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2)

majority_correct = sum(1 for r in final_results if r["pred_norag"] == r["correct_answer"])
print(f"\n=== No-RAG Baseline (majority vote, {n} questions) ===")
print(f"Accuracy: {majority_correct / n:.3f} ({majority_correct}/{n})")

# Also report average across the 3 individual runs for comparison with
# run_accuracy.py's per-run reporting style
per_run_acc = []
for r in range(N_RUNS):
    acc = sum(1 for x in all_run_results[r] if x["pred"] == x["correct_answer"]) / n
    per_run_acc.append(acc)
print(f"Per-run accuracy: {[f'{a:.3f}' for a in per_run_acc]}")
print(f"Mean ± std: {np.mean(per_run_acc):.3f} ± {np.std(per_run_acc):.3f}")