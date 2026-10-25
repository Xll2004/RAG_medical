# m2_medcpt.py —— 审稿意见 M2：换成医疗领域检索器 + 与考试题匹配的语料，看主要结论是否还成立
#   检索器：MedCPT（ncbi/MedCPT-Query-Encoder / ncbi/MedCPT-Article-Encoder），内积检索
#   语料：MedRAG textbooks（MedQA 配套的 18 本英文医学教科书切片，约 12.6 万段）
#   题干、选项、生成 prompt、3 次运行多数投票均与主实验相同；只换检索器和语料
# 用法：python -u m2_medcpt.py | tee m2.log      （第一次运行会下载模型和语料，并编码语料，可断点续跑）
from review2_common import *
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
EMB_DIR = "medcpt_textbooks_emb"
os.makedirs(EMB_DIR, exist_ok=True)
CACHE = "m2_cache.json"
cache = L(CACHE) if os.path.exists(CACHE) else {}
save = lambda: json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- 1. 语料 ----------
from datasets import load_dataset
ds = load_dataset("MedRAG/textbooks", split="train")
titles, contents = ds["title"], ds["content"]
texts = ds["contents"] if "contents" in ds.column_names else [f"{t}. {c}" for t, c in zip(titles, contents)]
print(f"textbooks: {len(texts)} snippets, device = {DEV}", flush=True)

# ---------- 2. 编码语料（分块存盘，可续跑）----------
CHUNK, BS = 4096, 16
n_chunks = (len(texts) + CHUNK - 1) // CHUNK
if not all(os.path.exists(f"{EMB_DIR}/{c:04d}.npy") for c in range(n_chunks)):
    atok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    amod = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(DEV).eval()
    for c in range(n_chunks):
        f = f"{EMB_DIR}/{c:04d}.npy"
        if os.path.exists(f):
            continue
        lo, hi = c * CHUNK, min((c + 1) * CHUNK, len(texts))
        embs = []
        for b in range(lo, hi, BS):
            pairs = [[titles[j], contents[j]] for j in range(b, min(b + BS, hi))]
            enc = atok(pairs, truncation=True, padding=True, return_tensors="pt", max_length=512).to(DEV)
            with torch.inference_mode():
                embs.append(amod(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
            del enc
            if DEV == "mps" and (b // BS) % 20 == 0:
                torch.mps.empty_cache()      # 不同长度的 padding 会让 MPS 缓存不断增长，定期释放
        np.save(f, np.concatenate(embs).astype("float32"))
        if DEV == "mps":
            torch.mps.empty_cache()
        print(f"encoded chunk {c+1}/{n_chunks}", flush=True)
    del amod
D = np.concatenate([np.load(f"{EMB_DIR}/{c:04d}.npy") for c in range(n_chunks)])
assert len(D) == len(texts)

# ---------- 3. 检索（内积，top-3）----------
if "docs_c1m" not in cache:
    qtok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    qmod = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(DEV).eval()
    def qenc(qs):
        out = []
        for b in range(0, len(qs), 32):
            enc = qtok(qs[b:b + 32], truncation=True, padding=True, return_tensors="pt", max_length=64).to(DEV)
            with torch.no_grad():
                out.append(qmod(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
        return np.concatenate(out)
    for qs, key in [(C1Q, "c1m"), (C2Q, "c2m")]:
        S = qenc(qs) @ D.T
        top = np.argsort(-S, axis=1)[:, :3]
        cache[f"idx_{key}"] = top.tolist()
        cache[f"docs_{key}"] = [[texts[j] for j in row] for row in top]
    save()
C1M, C2M = cache["docs_c1m"], cache["docs_c2m"]
RANDM = random_ctx([C1M, C2M])
J = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(C1M, C2M)]
print(f"MedCPT Jaccard(C1, C2) = {np.mean(J):.3f}; disjoint {sum(j == 0 for j in J)}/{N}", flush=True)

# ---------- 4. 生成 ----------
run_cells({
    "c1q_c1m": (C1Q, C1M), "c2q_c2m": (C2Q, C2M),       # 端到端 C1 / C2
    "c2q_c1m": (C2Q, C1M), "c1q_c2m": (C1Q, C2M),       # 2x2 分解
    "zhq_c1m": (ZHQ, C1M), "zhq_c2m": (ZHQ, C2M),       # 原题读者（交互）
    "c1q_randm": (C1Q, RANDM), "c2q_randm": (C2Q, RANDM), "zhq_randm": (ZHQ, RANDM),   # 随机段落对照
}, "results_m2_medcpt_200.json")
print("M2 done")
