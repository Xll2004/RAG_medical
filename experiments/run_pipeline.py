import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Loading index and abstracts...")
index = faiss.read_index("data/pubmed_index.faiss")
with open("data/pubmed_abstracts.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)

encoder = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/gold_set_200_translated.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

def retrieve(query, k=3):
    embedding = encoder.encode([query])
    D, I = index.search(np.array(embedding), k=k)
    return [abstracts[i]["text"] for i in I[0]]

def generate_answer(query, options_en, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    options_text = "\n".join([f"{k}: {v}" for k, v in options_en.items()])
    prompt = f"""You are a medical expert. Based on the following context, answer the question and select the most appropriate option.

Context:
{context}

Question: {query}

Options:
{options_text}

Provide a brief explanation and indicate which option is correct."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

print("Running Condition 2 (Machine Translation) on 200 questions...")
results = []

for i, q in enumerate(questions):
    retrieved = retrieve(q["question_mt"])
    answer = generate_answer(q["question_mt"], q["option_mt"], retrieved)

    results.append({
        "id": q["id"],
        "question_zh": q["question"],
        "question_mt": q["question_mt"],
        "option_mt": q["option_mt"],
        "retrieved_docs": retrieved,
        "answer": answer,
        "correct_answer": q["answer"]
    })

    if (i + 1) % 20 == 0:
        print(f"{i + 1}/200 done")

    time.sleep(0.3)

with open("results/results_condition2_200.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done. Saved to results_condition2_200.json")