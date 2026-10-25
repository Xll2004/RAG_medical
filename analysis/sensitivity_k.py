import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading index and abstracts...")
index = faiss.read_index("data/pubmed_index.faiss")
with open("data/pubmed_abstracts.json") as f:
    abstracts = json.load(f)

encoder = SentenceTransformer("all-MiniLM-L6-v2")

with open("results/results_condition1_200.json") as f:
    c1_data = json.load(f)
with open("results/results_condition2_200.json") as f:
    c2_data = json.load(f)

c1_queries = {q["id"]: q["question_en_baseline"] for q in c1_data}
c2_queries = {q["id"]: q["question_mt"] for q in c2_data}

def retrieve_ids(query, k):
    embedding = encoder.encode([query])
    D, I = index.search(np.array(embedding), k)
    return set(I[0].tolist())

for k in [3, 5, 10]:
    jaccard_scores = []
    zero_overlap = 0
    ids = list(c1_queries.keys())
    for qid in ids:
        s1 = retrieve_ids(c1_queries[qid], k)
        s2 = retrieve_ids(c2_queries[qid], k)
        intersection = s1 & s2
        union = s1 | s2
        jaccard = len(intersection) / len(union) if union else 0
        jaccard_scores.append(jaccard)
        if len(intersection) == 0:
            zero_overlap += 1
    avg = sum(jaccard_scores) / len(jaccard_scores)
    print(f"k={k}: avg Jaccard={avg:.3f}, zero overlap={zero_overlap} ({zero_overlap/len(ids)*100:.1f}%)")