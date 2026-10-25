import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from collections import Counter

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("results_condition1_200.json") as f:
    c1_data = json.load(f)
with open("results_condition2_200.json") as f:
    c2_data = json.load(f)

c1_map = {q["id"]: q for q in c1_data}
c2_map = {q["id"]: q for q in c2_data}

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

def run_condition(ids, get_query, get_docs, get_options, get_correct, label, output_file):
    results = []
    correct = 0
    n = len(ids)

    for i, qid in enumerate(ids):
        print(f"{label} Q{i+1}/{n}...", end=" ", flush=True)

        query = get_query(qid)
        docs = get_docs(qid)
        options = get_options(qid)
        correct_answer = get_correct(qid)

        prompt = build_prompt(query, docs, options)

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
        is_correct = int(final == correct_answer)
        correct += is_correct

        results.append({
            "id": qid,
            "answer": final,
            "correct_answer": correct_answer,
            "correct": is_correct
        })

        print(f"pred={final} gt={correct_answer} {'✓' if is_correct else '✗'}")

    accuracy = correct / n * 100
    print(f"\n=== {label} Accuracy: {accuracy:.1f}% ===")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {output_file}")
    return accuracy

ids = [q["id"] for q in c1_data]

# C1 query + C2 options（C1检索docs + C1 query + C2 options）
acc_c1q_c2o = run_condition(
    ids,
    get_query=lambda qid: c1_map[qid]["question_en_baseline"],
    get_docs=lambda qid: c1_map[qid]["retrieved_docs"],
    get_options=lambda qid: c2_map[qid]["option_mt"],
    get_correct=lambda qid: c1_map[qid]["correct_answer"],
    label="C1query_C2options",
    output_file="results_c1q_c2o_200.json"
)

# C2 query + C1 options（C2检索docs + C2 query + C1 options）
acc_c2q_c1o = run_condition(
    ids,
    get_query=lambda qid: c2_map[qid]["question_mt"],
    get_docs=lambda qid: c2_map[qid]["retrieved_docs"],
    get_options=lambda qid: c1_map[qid]["options_en"],
    get_correct=lambda qid: c2_map[qid]["correct_answer"],
    label="C2query_C1options",
    output_file="results_c2q_c1o_200.json"
)

print(f"\n=== Summary ===")
print(f"C1 (baseline):              55.0%")
print(f"C1 query + C2 options:      {acc_c1q_c2o:.1f}%")
print(f"C2 query + C1 options:      {acc_c2q_c1o:.1f}%")
print(f"C2 (MT baseline):           34.0%")