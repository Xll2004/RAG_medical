import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import time

load_dotenv()
gen_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ds_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

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

def correct_with_deepseek(text_mt):
    if not text_mt.strip():
        return text_mt
    response = ds_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": f"""You are a medical terminology expert.
Correct any medical terminology errors in this machine-translated text.
Reply with ONLY the corrected text, nothing else.

Machine-translated text: {text_mt}

Corrected text:"""}],
        max_tokens=200
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        result = content[0].strip() if content else text_mt
    else:
        result = content.strip() if content else text_mt
    for prefix in ["Corrected text:", "Corrected:", "Answer:"]:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
    return result if result else text_mt

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
    response = gen_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

print("Running Condition 4 (DeepSeek correction) on 200 questions...")
results = []

for i, q in enumerate(questions):
    question_corrected = correct_with_deepseek(q["question_mt"])
    time.sleep(0.2)

    options_corrected = {}
    for key, value in q["option_mt"].items():
        options_corrected[key] = correct_with_deepseek(value) if value.strip() else ""
        time.sleep(0.2)

    retrieved = retrieve(question_corrected)
    answer = generate_answer(question_corrected, options_corrected, retrieved)

    results.append({
        "id": q["id"],
        "question_zh": q["question"],
        "question_mt": q["question_mt"],
        "question_corrected": question_corrected,
        "options_corrected": options_corrected,
        "retrieved_docs": retrieved,
        "answer": answer,
        "correct_answer": q["answer"]
    })

    if (i + 1) % 20 == 0:
        print(f"{i + 1}/200 done")

    time.sleep(0.3)

with open("results/results_condition4_200.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done. Saved to results_condition4_200.json")