import json
import os
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Loading index and abstracts...")
index = faiss.read_index("data/pubmed_index_mminilm.faiss")
with open("data/pubmed_abstracts.json") as f:
    abstracts = json.load(f)

encoder_ml = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

with open("data/gold_set_200_translated.json") as f:
    questions = json.load(f)

with open("results/accuracy_results_200.json") as f:
    acc_data = json.load(f)
correct_answers = {q["id"]: q["correct_answer"] for q in acc_data}

def retrieve_multilingual(query_zh, k=3):
    embedding = encoder_ml.encode([query_zh], normalize_embeddings=True)
    D, I = index.search(np.array(embedding), k)
    return [abstracts[i]["text"] for i in I[0]]

def build_prompt(query, options, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"""You are a medical expert. Based on the following context, answer the question and select the most appropriate option.

Context:
{context}

Question: {query}

Options:
{options_text}

Provide a brief explanation and indicate which option is correct."""

def map_answer(generated, options):
    opts = "\n".join([f"{k}: {v}" for k, v in options.items()])
    prompt = f"""Select the closest matching option letter (A-E).

Generated answer: {generated}

Options:
{opts}

Reply with ONLY the option letter."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0
    )
    return response.choices[0].message.content.strip()

results = []
correct = 0
n = len(questions)

for i, q in enumerate(questions):
    print(f"Q{i+1}/{n}...", end=" ", flush=True)

    docs = retrieve_multilingual(q["question"])

    prompt = build_prompt(q["question"], q["option"], docs)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    pred = map_answer(raw, q["option"])

    correct_answer = correct_answers[q["id"]]
    is_correct = int(pred == correct_answer)
    correct += is_correct

    results.append({
        "id": q["id"],
        "answer": pred,
        "correct_answer": correct_answer,
        "correct": is_correct
    })

    print(f"pred={pred} gt={correct_answer} {'✓' if is_correct else '✗'}")

accuracy = correct / n * 100
print(f"\n=== Multilingual Embedding (C0) Accuracy: {accuracy:.1f}% ===")

with open("results/results_c0_mminilm_fixed_200.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results_c0_mminilm_fixed_200.json")