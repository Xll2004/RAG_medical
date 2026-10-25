import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from dotenv import load_dotenv
from openai import OpenAI
import time
import os
from collections import Counter

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_answer_choice(question, options, retrieved_docs):
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
    return response.choices[0].message.content.strip()[0].upper()

# Load data
with open("results/results_semantic_t3.json") as f:
    semantic = json.load(f)
with open("results/results_condition2_200.json") as f:
    c2_all = json.load(f)
with open("results/results_condition3_200.json") as f:
    c3_all = json.load(f)
with open("results/results_condition5_200.json") as f:
    c5_all = json.load(f)
with open("data/gold_set_200_translated.json") as f:
    questions = json.load(f)

c2_by_id = {r["id"]: r for r in c2_all}
c3_by_id = {r["id"]: r for r in c3_all}
c5_by_id = {r["id"]: r for r in c5_all}
q_by_id = {q["id"]: q for q in questions}

print(f"Evaluating {len(semantic)} T3 cases...")

# Run 3 times for majority vote
N_RUNS = 3
all_run_results = []

for run in range(N_RUNS):
    print(f"\nRun {run+1}/{N_RUNS}...")
    run_results = []
    for i, s in enumerate(semantic):
        qid = s["id"]
        q = q_by_id.get(qid)
        if not q:
            continue
        pred_c2 = get_answer_choice(c2_by_id[qid]["question_mt"], q["option"], c2_by_id[qid]["retrieved_docs"])
        time.sleep(0.2)
        pred_c3 = get_answer_choice(c3_by_id[qid]["question_corrected"], q["option"], c3_by_id[qid]["retrieved_docs"])
        time.sleep(0.2)
        pred_c5 = get_answer_choice(c5_by_id[qid]["question_corrected"], q["option"], c5_by_id[qid]["retrieved_docs"])
        time.sleep(0.2)
        pred_sem = get_answer_choice(s["question_semantic"], q["option"], s["retrieved_docs"])
        time.sleep(0.2)

        run_results.append({
            "id": qid,
            "correct_answer": s["correct_answer"],
            "pred_c2": pred_c2,
            "pred_c3": pred_c3,
            "pred_c5": pred_c5,
            "pred_sem": pred_sem
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(semantic)} done")

    all_run_results.append(run_results)

# Majority vote
n = len(semantic)
final = []
for i in range(n):
    item = {
        "id": all_run_results[0][i]["id"],
        "correct_answer": all_run_results[0][i]["correct_answer"]
    }
    for k in ["pred_c2", "pred_c3", "pred_c5", "pred_sem"]:
        votes = [all_run_results[r][i][k] for r in range(N_RUNS)]
        item[k] = Counter(votes).most_common(1)[0][0]
    final.append(item)

# Calculate accuracy
c2 = [int(r["pred_c2"] == r["correct_answer"]) for r in final]
c3 = [int(r["pred_c3"] == r["correct_answer"]) for r in final]
c5 = [int(r["pred_c5"] == r["correct_answer"]) for r in final]
sem = [int(r["pred_sem"] == r["correct_answer"]) for r in final]

print(f"\n=== Final Results on {n} T3 Cases (majority vote) ===")
print(f"C2 MT:                 {np.mean(c2):.3f} ({sum(c2)}/{n})")
print(f"C3 General:            {np.mean(c3):.3f} ({sum(c3)}/{n})")
print(f"C5 GPT-5.5:            {np.mean(c5):.3f} ({sum(c5)}/{n})")
print(f"Semantic Preservation: {np.mean(sem):.3f} ({sum(sem)}/{n})")

# McNemar tests
def run_mcnemar(a, b, label):
    bc = sum(1 for x,y in zip(a,b) if x==1 and y==1)
    ao = sum(1 for x,y in zip(a,b) if x==1 and y==0)
    bo = sum(1 for x,y in zip(a,b) if x==0 and y==1)
    bw = sum(1 for x,y in zip(a,b) if x==0 and y==0)
    result = mcnemar([[bc,ao],[bo,bw]], exact=True)
    sig = "✅ significant" if result.pvalue < 0.05 else "❌ not significant"
    print(f"{label}: p = {result.pvalue:.4f} {sig}")

print("\n=== McNemar Tests ===")
run_mcnemar(c2, sem, "C2 vs Semantic Preservation")
run_mcnemar(c3, sem, "C3 vs Semantic Preservation")
run_mcnemar(c5, sem, "C5 vs Semantic Preservation")
run_mcnemar(c2, c3,  "C2 vs C3")
run_mcnemar(c2, c5,  "C2 vs C5")