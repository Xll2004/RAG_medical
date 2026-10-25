import json
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from bert_score import score as bert_score

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30)

def build_prompt(question, options, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"""You are a medical expert. Based on the following context, answer the question and select the correct option.

Context:
{context}

Question: {question}

Options:
{options_text}

Provide a brief explanation and indicate which option is correct."""

def get_sample(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=1.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Sample error: {e}")
        return ""

def selfcheck_bertscore(question, options, answer, retrieved_docs, n_samples=3):
    prompt = build_prompt(question, options, retrieved_docs)
    samples = [get_sample(prompt) for _ in range(n_samples)]
    samples = [s for s in samples if s]
    if not samples:
        return 0.5
    references = [answer] * len(samples)
    P, R, F1 = bert_score(samples, references, lang="en", verbose=False)
    return 1 - F1.mean().item()

# Load results
with open("results_condition1_200.json") as f: c1 = json.load(f)
with open("results_condition2_200.json") as f: c2 = json.load(f)
with open("results_condition3_200.json") as f: c3 = json.load(f)
with open("results_condition4_200.json") as f: c4 = json.load(f)
with open("results_condition5_200.json") as f: c5 = json.load(f)

c2_opt_key = "option_mt" if "option_mt" in c2[0] else "options_mt"

n = min(len(c1), len(c2), len(c3), len(c4), len(c5), 200)
misaligned = sum(1 for i in range(n) if not (
    c1[i]["id"] == c2[i]["id"] == c3[i]["id"] == c4[i]["id"] == c5[i]["id"]
))
print(f"Using {n} questions, ID misalignments: {misaligned}")

print("Running SelfCheckGPT (BERTScore) on 200 questions...")
scores = []

for i in range(n):
    print(f"Q{i+1}/{n}...", end=" ", flush=True)

    s1 = selfcheck_bertscore(c1[i]["question_en_baseline"], c1[i]["options_en"], c1[i]["answer"], c1[i]["retrieved_docs"])
    s2 = selfcheck_bertscore(c2[i]["question_mt"], c2[i][c2_opt_key], c2[i]["answer"], c2[i]["retrieved_docs"])
    s3 = selfcheck_bertscore(c3[i]["question_corrected"], c3[i]["options_corrected"], c3[i]["answer"], c3[i]["retrieved_docs"])
    s4 = selfcheck_bertscore(c4[i]["question_corrected"], c4[i]["options_corrected"], c4[i]["answer"], c4[i]["retrieved_docs"])
    s5 = selfcheck_bertscore(c5[i]["question_corrected"], c5[i]["options_corrected"], c5[i]["answer"], c5[i]["retrieved_docs"])

    scores.append({
        "id": c1[i]["id"],
        "score_c1": s1, "score_c2": s2, "score_c3": s3,
        "score_c4": s4, "score_c5": s5
    })

    print(f"C1:{s1:.2f} C2:{s2:.2f} C3:{s3:.2f} C4:{s4:.2f} C5:{s5:.2f}")

    with open("selfcheck_scores_bertscore_200.json", "w") as f:
        json.dump(scores, f, indent=2)

print("\n=== Final Results (BERTScore 200 questions) ===")
for j in range(1, 6):
    avg = np.mean([s[f"score_c{j}"] for s in scores])
    print(f"Condition {j}: {avg:.3f}")