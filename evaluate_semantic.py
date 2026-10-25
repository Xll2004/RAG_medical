import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import time
import os

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

# Load semantic results
with open("results_semantic_t3.json") as f:
    semantic = json.load(f)

# Load other conditions for same IDs
with open("results_condition2_200.json") as f:
    c2_all = json.load(f)
with open("results_condition3_200.json") as f:
    c3_all = json.load(f)
with open("results_condition5_200.json") as f:
    c5_all = json.load(f)
with open("gold_set_200_translated.json") as f:
    questions = json.load(f)

# Build lookups by ID
target_ids = [r["id"] for r in semantic]
c2_by_id = {r["id"]: r for r in c2_all}
c3_by_id = {r["id"]: r for r in c3_all}
c5_by_id = {r["id"]: r for r in c5_all}
q_by_id = {q["id"]: q for q in questions}

print(f"Evaluating {len(semantic)} T3 cases...")
print("Running 3 times for stability...")

N_RUNS = 3
all_runs = {k: [] for k in ["c2", "c3", "c5", "semantic"]}

for run in range(N_RUNS):
    print(f"\n--- Run {run+1}/{N_RUNS} ---")
    correct = {k: 0 for k in ["c2", "c3", "c5", "semantic"]}

    for i, s in enumerate(semantic):
        qid = s["id"]
        q = q_by_id.get(qid)
        if not q:
            continue
        correct_ans = s["correct_answer"]

        pred_c2 = get_answer_choice(c2_by_id[qid]["question_mt"], q["option"], c2_by_id[qid]["retrieved_docs"])
        time.sleep(0.2)
        pred_c3 = get_answer_choice(c3_by_id[qid]["question_corrected"], q["option"], c3_by_id[qid]["retrieved_docs"])
        time.sleep(0.2)
        pred_c5 = get_answer_choice(c5_by_id[qid]["question_corrected"], q["option"], c5_by_id[qid]["retrieved_docs"])
        time.sleep(0.2)
        pred_sem = get_answer_choice(s["question_semantic"], q["option"], s["retrieved_docs"])
        time.sleep(0.2)

        correct["c2"] += int(pred_c2 == correct_ans)
        correct["c3"] += int(pred_c3 == correct_ans)
        correct["c5"] += int(pred_c5 == correct_ans)
        correct["semantic"] += int(pred_sem == correct_ans)

        if (i + 1) % 5 == 0:
            n = i + 1
            print(f"  {n}/{len(semantic)} | C2:{correct['c2']/n:.3f} C3:{correct['c3']/n:.3f} C5:{correct['c5']/n:.3f} Sem:{correct['semantic']/n:.3f}")

    for k in all_runs:
        all_runs[k].append(correct[k])

    print(f"Run {run+1}: C2={correct['c2']/len(semantic):.3f} C3={correct['c3']/len(semantic):.3f} C5={correct['c5']/len(semantic):.3f} Semantic={correct['semantic']/len(semantic):.3f}")

n = len(semantic)
print(f"\n=== Final Results on {n} T3 Cases (3-run average) ===")
for k, label in [("c2", "C2 MT"), ("c3", "C3 General"), ("c5", "C5 GPT-5.5"), ("semantic", "Semantic Preservation")]:
    avg = np.mean(all_runs[k]) / n
    std = np.std([r/n for r in all_runs[k]])
    print(f"{label:<25}: {avg:.3f} ± {std:.3f}")