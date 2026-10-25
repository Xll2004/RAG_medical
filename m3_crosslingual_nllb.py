# m3_crosslingual_nllb.py —— 审稿意见 M3
#  (a) C0：跨语言 dense retriever，中文原题直接检索英文语料（paraphrase-multilingual-MiniLM-L12-v2 +
#      已有的 pubmed_index_mminilm.faiss），完全绕过翻译；生成器读中文原题
#  (b) C6：中等质量 MT（NLLB-200 distilled 600M），检索/生成与 C2 完全相同，用来把 C1 vs C2 变成质量梯度
# 用法：python -u m3_crosslingual_nllb.py | tee m3.log
from review2_common import *          # 先设好 OMP 环境变量
import numpy as np
import torch

CACHE = "m3_cache.json"
cache = L(CACHE) if os.path.exists(CACHE) else {}
save = lambda: json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- 1. NLLB 翻译 ----------
if "q_c6" not in cache:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="zho_Hans")
    mt = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    out = []
    for i in range(0, N, 8):
        batch = ZHQ[i:i + 8]
        inp = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            gen = mt.generate(**inp, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
                              max_length=512, num_beams=4)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        print(f"NLLB {min(i + 8, N)}/{N}", flush=True)
    cache["q_c6"] = out
    save()
    del mt, tok
C6Q = cache["q_c6"]

# ---------- 2. 检索：先用 torch 编码全部 query，再加载 FAISS（避免 macOS OpenMP 冲突）----------
if "docs_c0" not in cache or "docs_c6" not in cache:
    from sentence_transformers import SentenceTransformer
    e_ml = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    E0 = np.ascontiguousarray(e_ml.encode(ZHQ, normalize_embeddings=True), dtype="float32")   # 与 run_c0_mminilm_fixed.py 一致
    del e_ml
    e_en = SentenceTransformer("all-MiniLM-L6-v2")
    E6 = np.ascontiguousarray(e_en.encode(C6Q), dtype="float32")                              # 与 run_condition1.py 一致
    del e_en
    import faiss
    faiss.omp_set_num_threads(1)
    abstracts = L("pubmed_abstracts.json")
    for path, E, key in [("pubmed_index_mminilm.faiss", E0, "docs_c0"), ("pubmed_index.faiss", E6, "docs_c6")]:
        print("Loading", path, flush=True)
        index = faiss.read_index(path)
        _, I = index.search(E, 3)
        cache[key] = [[abstracts[j]["text"] for j in row] for row in I]
        del index
    del abstracts
    save()
C0DOCS, C6DOCS = cache["docs_c0"], cache["docs_c6"]

# ---------- 3. 生成（3 次运行 + 多数投票）----------
run_cells({
    "zhq_c0docs": (ZHQ, C0DOCS),     # 跨语言检索 + 读原题：不翻译的完整流水线
    "c6q_c6docs": (C6Q, C6DOCS),     # NLLB 端到端
    "c6q_c1docs": (C6Q, C1DOCS),     # NLLB 题干 + 忠实证据（分解用）
    "zhq_c6docs": (ZHQ, C6DOCS),     # 原题 + NLLB 证据（交互用）
}, "results_m3_200.json")

# ---------- 4. NLLB 译文的 GEMBA 质量分（与 exp3_qe.py 同一 prompt；Claude 主评分、GPT 第二评分）----------
import re
from llm_client import LLM
GEMBA = ('Score the following translation from Chinese to English on a continuous scale from 0 to 100, '
         'where score of zero means "no meaning preserved" and score of one hundred means '
         '"perfect meaning and grammar".\n\nChinese source: "{src}"\nEnglish translation: "{tgt}"\nScore: ')
SYSTEM = "Reply with only the score, a number from 0 to 100."
for prov in ["claude", "gpt"]:
    key = f"qe_c6_{prov}"
    if key in cache:
        continue
    llm = LLM(prov)
    def score(i):
        msgs = [{"role": "user", "content": GEMBA.format(src=ZHQ[i], tgt=C6Q[i])}]
        raw = llm.chat(msgs, system=SYSTEM, max_tokens=10)
        m = re.match(r"\s*\**(\d+(?:\.\d+)?)", raw)
        if m:
            return float(m.group(1))
        raw = llm.chat(msgs, system=SYSTEM, max_tokens=1024)
        ms = re.findall(r"(?i)score\W{0,5}(\d+(?:\.\d+)?)", raw) or re.findall(r"\b(\d+(?:\.\d+)?)\b", raw)
        return float(ms[-1]) if ms else None
    with ThreadPoolExecutor(6) as ex:
        cache[key] = list(ex.map(score, range(N)))
    save()
    s = [x for x in cache[key] if x is not None]
    print(f"QE C6 ({prov}): mean {sum(s)/len(s):.1f}", flush=True)
print("M3 done")
