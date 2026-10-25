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

# Load all data
with open("gold_set_200_translated.json", "r", encoding="utf-8") as f:
    questions = json.load(f)
with open("results_condition1_200.json", "r", encoding="utf-8") as f:
    c1 = json.load(f)
with open("results_condition2_200.json", "r", encoding="utf-8") as f:
    c2 = json.load(f)
with open("results_condition3_200.json", "r", encoding="utf-8") as f:
    c3 = json.load(f)
with open("results_condition4_200.json", "r", encoding="utf-8") as f:
    c4 = json.load(f)
with open("results_condition5_200.json", "r", encoding="utf-8") as f:
    c5 = json.load(f)

conditions = [
    (c1, "question_en_baseline"),
    (c2, "question_mt"),
    (c3, "question_corrected"),
    (c4, "question_corrected"),
    (c5, "question_corrected")
]

# Verify alignment
n = min(len(c1), len(c2), len(c3), len(c4), len(c5), 200)
misaligned = sum(1 for i in range(n) if not (
    c1[i]["id"] == c2[i]["id"] == c3[i]["id"] == c4[i]["id"] == c5[i]["id"]
))
gold_misaligned = sum(1 for i in range(n) if questions[i]["id"] != c1[i]["id"])
print(f"Using {n} questions, ID misalignments across conditions: {misaligned}, gold vs results: {gold_misaligned}")

# Run 3 times and average
N_RUNS = 3
all_results = []
all_run_results = []

for run in range(N_RUNS):
    print(f"\n--- Run {run+1}/{N_RUNS} ---")
    correct = [0] * 5
    run_results = []

    for i, q in enumerate(questions):
        preds = []
        for c, key in conditions:
            pred = get_answer_choice(c[i][key], q["option"], c[i]["retrieved_docs"])
            preds.append(pred)
            time.sleep(0.2)

        ans = q["answer"]
        for j in range(5):
            correct[j] += int(preds[j] == ans)

        run_results.append({
            "id": q["id"],
            "correct_answer": ans,
            "preds": preds
        })

        if (i + 1) % 40 == 0:
            print(f"  {i+1}/200 | " + " | ".join([f"C{j+1}: {correct[j]/(i+1):.3f}" for j in range(5)]))

        time.sleep(0.2)

    all_results.append(correct)
    all_run_results.append(run_results)
    print(f"Run {run+1}: " + " | ".join([f"C{j+1}: {correct[j]/200:.3f}" for j in range(5)]))

# Majority vote
final_accuracy_results = []
for i in range(n):
    item = {
        "id": all_run_results[0][i]["id"],
        "correct_answer": all_run_results[0][i]["correct_answer"]
    }
    for j, cname in enumerate(["c1", "c2", "c3", "c4", "c5"]):
        votes = [all_run_results[r][i]["preds"][j] for r in range(N_RUNS)]
        item[f"pred_{cname}"] = Counter(votes).most_common(1)[0][0]
    final_accuracy_results.append(item)

with open("accuracy_results_200.json", "w", encoding="utf-8") as f:
    json.dump(final_accuracy_results, f, indent=2)

print("\n=== Final Averaged Results (3 runs, 200 questions) ===")
for j in range(5):
    avg = sum(r[j] for r in all_results) / (N_RUNS * 200)
    std = np.std([r[j]/200 for r in all_results])
    print(f"Condition {j+1}: {avg:.3f} ± {std:.3f}")