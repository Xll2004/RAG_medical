import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load index and abstracts
print("Loading index and abstracts...")
index = faiss.read_index("pubmed_index.faiss")
with open("pubmed_abstracts.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)

# Load encoder
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Load gold set with translations
with open("gold_set_200_translated.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

def retrieve(query, k=3):
    """Retrieve top-k relevant abstracts for a given query."""
    embedding = encoder.encode([query])
    D, I = index.search(np.array(embedding), k=k)
    return [abstracts[i]["text"] for i in I[0]]

def build_prompt(query, retrieved_docs):
    """Build RAG prompt from query and retrieved documents."""
    context = "\n\n".join(retrieved_docs)
    prompt = f"""You are a medical expert. Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    return prompt

if __name__ == "__main__":
    # Test on first question - Condition 2 (machine translation)
    q = questions[0]
    print(f"Original (Chinese): {q['question']}")
    print(f"Translated (MT):    {q['question_mt']}")
    print()

    retrieved = retrieve(q["question_mt"])
    prompt = build_prompt(q["question_mt"], retrieved)
    print("--- Generated Prompt ---")
    print(prompt)