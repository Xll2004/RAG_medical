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

def translate_gpt41(text_zh):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": f"Translate this Chinese medical text to English accurately, preserving all medical terminology:\n\n{text_zh}"}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

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

print("Running C1' (GPT-4.1 translation) on 200 questions...")
results = []

for i, q in enumerate(questions):
    question_en = translate_gpt41(q["question"])
    time.sleep(0.2)

    options_en = {}
    for key, value in q["option"].items():
        if value.strip():
            options_en[key] = translate_gpt41(value)
            time.sleep(0.2)
        else:
            options_en[key] = ""

    retrieved = retrieve(question_en)
    answer = generate_answer(question_en, options_en, retrieved)

    results.append({
        "id": q["id"],
        "question_zh": q["question"],
        "question_en_gpt41": question_en,
        "options_en": options_en,
        "retrieved_docs": retrieved,
        "answer": answer,
        "correct_answer": q["answer"]
    })

    if (i + 1) % 20 == 0:
        print(f"{i+1}/200 done")

    time.sleep(0.3)

with open("results/results_c1prime_200.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done. Saved results_c1prime_200.json")