import json
import os
import time
import numpy as np
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from statsmodels.stats.contingency_tables import mcnemar

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_answer_choice(question, options, retrieved_docs):
    """Identical to run_accuracy.py's get_answer_choice."""
    context = "\n\n".join(retrieved_docs)
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    prompt = f"""You are a medical expert. Based on the context below, select the correct answer.
Only reply with a single letter (A, B, C, D, or E).
Context:
{context}
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


with open("gold_set_200_translated.json", "r", encoding="utf-8") as f:
    questions = json.load(f)
with open("results_c1prime_200.json", "r", encoding="utf-8") as f:
    c1p = json.load(f)

n = min(len(questions), len(c1p), 200)
misaligned = sum(1 for i in range(n) if questions[i]["id"] != c1p[i]["id"])
print(f"Using {n} questions, ID misalignments: {misaligned}")

N_RUNS = 3
all_run_results = []

for run in range(N_RUNS):
    print(f"\n--- Run {run + 1}/{N_RUNS} ---")
    correct = 0
    run_results = []
    for i, q in enumerate(questions[:n]):
        pred = get_answer_choice(
            c1p[i]["question_en_gpt41"],
            q["option"],
            c1p[i]["retrieved_docs"]
        )
        ans = q["answer"]
        correct += int(pred == ans)
        run_results.append({"id": q["id"], "correct_answer": ans, "pred": pred})
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{n} | C1' acc so far: {correct / (i + 1):.3f}")
        time.sleep(0.2)
    all_run_results.append(run_results)
    print(f"Run {run + 1}: C1' acc = {correct / n:.3f}")

# Majority vote
final = []
for i in range(n):
    votes = [all_run_results[r][i]["pred"] for r in range(N_RUNS)]
    final.append({
        "id": all_run_results[0][i]["id"],
        "correct_answer": all_run_results[0][i]["correct_answer"],
        "pred_c1prime": Counter(votes).most_common(1)[0][0]
    })

with open("accuracy_c1prime_200.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2)

acc = sum(1 for r in final if r["pred_c1prime"] == r["correct_answer"]) / n
print(f"\n=== C1' (GPT-4.1 translation) ===")
print(f"Accuracy: {acc:.3f}")

# McNemar vs C1
with open("accuracy_results_200.json") as f:
    main_results = json.load(f)
main_by_id = {r["id"]: r for r in main_results}
c1p_by_id = {r["id"]: r for r in final}
ids = [i for i in main_by_id if i in c1p_by_id]

c1 = [int(main_by_id[i]["pred_c1"] == main_by_id[i]["correct_answer"]) for i in ids]
c1prime = [int(c1p_by_id[i]["pred_c1prime"] == c1p_by_id[i]["correct_answer"]) for i in ids]

print(f"\nC1:  {np.mean(c1):.3f}")
print(f"C1': {np.mean(c1prime):.3f}")

both = sum(1 for x, y in zip(c1prime, c1) if x == 1 and y == 1)
a_only = sum(1 for x, y in zip(c1prime, c1) if x == 1 and y == 0)
b_only = sum(1 for x, y in zip(c1prime, c1) if x == 0 and y == 1)
neither = sum(1 for x, y in zip(c1prime, c1) if x == 0 and y == 0)
res = mcnemar([[both, a_only], [b_only, neither]], exact=False)
sig = "significant" if res.pvalue < 0.05 else "not significant"
print(f"\nMcNemar C1' vs C1: p = {res.pvalue:.4f} ({sig})")
print(f"  discordant: C1'-only correct = {a_only}, C1-only correct = {b_only}")