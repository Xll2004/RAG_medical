import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from collections import Counter

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load C2's retrieved docs (fixed retrieval results)
with open("results/results_condition2_200.json") as f:
    c2_data = json.load(f)
c2_docs = {q["id"]: q["retrieved_docs"] for q in c2_data}

# Load C3's corrected query
with open("results/results_condition3_200.json") as f:
    c3_data = json.load(f)

# Load C5's corrected query
with open("results/results_condition5_200.json") as f:
    c5_data = json.load(f)

def build_prompt(query, retrieved_docs, options):
    context = "\n\n".join(retrieved_docs)
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"""You are a medical expert. Based on the following context, answer the multiple choice question by selecting the best option.

Context:
{context}

Question: {query}

Options:
{options_text}

Answer with the option letter only (A, B, C, D, or E)."""

def map_answer(generated, options):
    prompt = f"""Given this generated answer, select the closest matching option letter (A-E).

Generated answer: {generated}

Options:
{chr(10).join([f"{k}: {v}" for k, v in options.items()])}

Reply with ONLY the option letter."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def run_condition(data, c2_docs, condition_name, output_file):
    results = []
    n = len(data)
    correct = 0

    for i, q in enumerate(data):
        qid = q["id"]
        print(f"{condition_name} Q{i+1}/{n}...", end=" ", flush=True)

        query = q["question_corrected"]
        options = q["options_corrected"]
        docs = c2_docs[qid]  # Force use of C2 retrieval results

        prompt = build_prompt(query, docs, options)

        # Three-run majority vote
        votes = []
        for _ in range(3):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0
            )
            raw = response.choices[0].message.content.strip()
            mapped = map_answer(raw, options)
            votes.append(mapped)

        final = Counter(votes).most_common(1)[0][0]
        is_correct = int(final == q["correct_answer"])
        correct += is_correct

        results.append({
            "id": qid,
            "question_corrected": query,
            "retrieved_docs": docs,
            "votes": votes,
            "answer": final,
            "correct_answer": q["correct_answer"],
            "correct": is_correct
        })

        print(f"pred={final} gt={q['correct_answer']} {'✓' if is_correct else '✗'}")

    accuracy = correct / n * 100
    print(f"\n=== {condition_name} Accuracy: {accuracy:.1f}% ===")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {output_file}")
    return accuracy

# Run C2a_C3
acc_c3 = run_condition(c3_data, c2_docs, "C2a_C3", "results/results_c2a_c3_200.json")

# Run C2a_C5
acc_c5 = run_condition(c5_data, c2_docs, "C2a_C5", "results/results_c2a_c5_200.json")

print(f"\n=== Summary ===")
print(f"C2a_C3 (C3 query + C2 docs): {acc_c3:.1f}%")
print(f"C2a_C5 (C5 query + C2 docs): {acc_c5:.1f}%")
print(f"Compare with C3 (C3 query + C3 docs): 38.0%")
print(f"Compare with C5 (C5 query + C5 docs): 44.0%")
print(f"Compare with C2 baseline: 34.0%")