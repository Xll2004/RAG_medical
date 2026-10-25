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

print("Loading index and abstracts...")
index = faiss.read_index("pubmed_index.faiss")
with open("pubmed_abstracts.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)

encoder = SentenceTransformer("all-MiniLM-L6-v2")

with open("gold_set_200_translated.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

with open("error_annotations_200.json") as f:
    errors = json.load(f)

# Get top 50 T3 cases
t3_ids = set([e["id"] for e in errors if "3" in e["error_types"]][:50])
target_questions = [q for q in questions if q["id"] in t3_ids]
print(f"Testing on {len(target_questions)} T3 cases")

def retrieve(query, k=3):
    embedding = encoder.encode([query])
    D, I = index.search(np.array(embedding), k=k)
    return [abstracts[i]["text"] for i in I[0]]

def correct_semantic(text_mt, original_zh):
    if not text_mt.strip():
        return text_mt
    response = gen_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""You are a medical translation expert.
The following is a machine-translated English version of a Chinese medical text.
The original Chinese is provided for reference.
Your task is to rewrite the English translation to accurately preserve the medical meaning of the original Chinese.
Focus on correcting semantic drift and ensuring the overall meaning is correct.
Reply with ONLY the corrected English text, nothing else.

Original Chinese: {original_zh}
Machine translation: {text_mt}

Semantically corrected translation:"""}],
        max_tokens=300
    )
    result = response.choices[0].message.content.strip()
    for prefix in ["Corrected translation:", "Corrected:", "Answer:"]:
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

print("Running Semantic Preservation Condition on T3 cases...")
results = []

for i, q in enumerate(target_questions):
    question_corrected = correct_semantic(q["question_mt"], q["question"])
    time.sleep(0.2)

    options_corrected = {}
    for key, value in q["option_mt"].items():
        if value.strip():
            zh_option = q["option"].get(key, "")
            options_corrected[key] = correct_semantic(value, zh_option)
            time.sleep(0.2)
        else:
            options_corrected[key] = ""

    retrieved = retrieve(question_corrected)
    answer = generate_answer(question_corrected, options_corrected, retrieved)

    results.append({
        "id": q["id"],
        "question_zh": q["question"],
        "question_mt": q["question_mt"],
        "question_semantic": question_corrected,
        "options_corrected": options_corrected,
        "retrieved_docs": retrieved,
        "answer": answer,
        "correct_answer": q["answer"]
    })

    if (i + 1) % 5 == 0:
        print(f"{i+1}/{len(target_questions)} done")

    time.sleep(0.3)

with open("results_semantic_t3.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done. Saved to results_semantic_t3.json")