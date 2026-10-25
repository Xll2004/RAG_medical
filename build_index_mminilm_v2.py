# build_index_mminilm_v2.py
# 从检索脚本实际使用的 pubmed_abstracts.json 构建 mMiniLM 索引，保证下标与旧索引一一对应
import json
import faiss
from sentence_transformers import SentenceTransformer

with open("pubmed_abstracts.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)
texts = [a["text"] for a in abstracts]
print(f"Loaded {len(texts)} texts")
assert len(texts) == 852618, "数量与旧索引不一致，停止"

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
faiss.write_index(index, "pubmed_index_mminilm.faiss")
print(f"Saved pubmed_index_mminilm.faiss ({index.ntotal} vectors, dim={index.d})")
