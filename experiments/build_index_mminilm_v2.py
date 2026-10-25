# build_index_mminilm_v2.py
# Build the mMiniLM index from the pubmed_abstracts.json the retrieval scripts actually use, so indices map one-to-one to the old index
import json
import faiss
from sentence_transformers import SentenceTransformer

with open("data/pubmed_abstracts.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)
texts = [a["text"] for a in abstracts]
print(f"Loaded {len(texts)} texts")
assert len(texts) == 852618, "Count differs from the old index, stopping"

encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("max_seq_length =", encoder.max_seq_length)

embeddings = encoder.encode(
    texts,
    show_progress_bar=True,
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "data/pubmed_index_mminilm.faiss")
print(f"Saved pubmed_index_mminilm.faiss ({index.ntotal} vectors, dim={index.d})")
