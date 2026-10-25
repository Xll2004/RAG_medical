import json
import os
import time
import numpy as np
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

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
    text = response.choices[0].message.content.strip()
    return text[0].upper() if text else "X"

# Load data
with open("gold_set_200_translated.json", "r", encoding="utf-8") as f:
    questions = json.load(f)
with open("results_condition3b_200.json", "r", encoding="utf-8") as f:
    c3b = json.load(f)

# Run 3 times and average
N_RUNS = 3
all_results = []
all_run_results = []

for run in range(N_RUNS):
    print(f"\n--- Run {run+1}/{N_RUNS} ---")
    correct = 0
    run_results = []

    for i, q in enumerate(questions):
        pred = get_answer_choice(
            c3b[i]["question_corrected"],
            q["option"],
            c3b[i]["retrieved_docs"]
        )
        is_correct = int(pred == q["answer"])
        correct += is_correct

        run_results.append({
            "id": q["id"],
            "correct_answer": q["answer"],
            "pred_c3b": pred
        })

        if (i + 1) % 40 == 0:
            print(f"  {i+1}/200 | Acc: {correct/(i+1):.3f}")

        time.sleep(0.2)

    all_results.append(correct)
    all_run_results.append(run_results)
    print(f"Run {run+1}: {correct/200:.3f}")

# Save majority vote
final_results = []
for i in range(200):
    votes = [all_run_results[r][i]["pred_c3b"] for r in range(N_RUNS)]
    final_results.append({
        "id": all_run_results[0][i]["id"],
        "correct_answer": all_run_results[0][i]["correct_answer"],
        "pred_c3b": Counter(votes).most_common(1)[0][0]
    })

with open("accuracy_results_c3b_200.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2)

avg = np.mean(all_results) / 200
std = np.std([r/200 for r in all_results])
print(f"\n=== Final Results (C3b, 200 questions) ===")
print(f"Condition 3b (focused terminology): {avg:.3f} ± {std:.3f}")

# Compare with C3
with open("accuracy_results_200.json") as f:
    acc_200 = json.load(f)
c2_acc = np.mean([int(r["pred_c2"] == r["correct_answer"]) for r in acc_200])
c3_acc = np.mean([int(r["pred_c3"] == r["correct_answer"]) for r in acc_200])
print(f"\nComparison:")
print(f"C2 MT:           {c2_acc:.3f}")
print(f"C3 General:      {c3_acc:.3f}")
print(f"C3b Terminology: {avg:.3f}")