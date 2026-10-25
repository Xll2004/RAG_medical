# run_pubmedqa_transfer.py
# 检索有用 + 有 gold evidence 的迁移实验：PubMedQA (pqa_labeled) 200 题
# 流程：EN 原题 -> GPT-4o-mini 译成中文（模拟中文用户提问）
#      -> C1: GPT-4o-mini 回译（与主实验 translate_to_english_baseline 同 prompt）
#      -> C2: opus-mt-zh-en（与主实验 translate.py 同模型）
#      -> 同一 FAISS 索引 + all-MiniLM-L6-v2 检索 top-3，按 pubid 计算 gold hit@3
#      -> 7 个格子 × 3 次运行，temperature=0，多数投票
# 支持断点续跑：所有中间结果缓存在 pubmedqa_transfer_cache.json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, random
from collections import Counter
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CACHE = "pubmedqa_transfer_cache.json"
OUT = "results_pubmedqa_transfer_200.json"
N, SEED, K, RUNS = 200, 42, 3, 3

cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
def save():
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- 1. 抽样 ----------
if "items" not in cache:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    idx = sorted(random.Random(SEED).sample(range(len(ds)), N))
    cache["items"] = [{
        "pubid": int(ds[i]["pubid"]),
        "question_en": ds[i]["question"],
        "gold_context": "\n\n".join(ds[i]["context"]["contexts"]),
        "answer": ds[i]["final_decision"],          # yes / no / maybe
    } for i in idx]
    save()
items = cache["items"]
print("label dist:", Counter(x["answer"] for x in items))

# ---------- 2. EN -> ZH（模拟中文提问） ----------
for x in items:
    if "question_zh" not in x:
        r = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0, max_tokens=300,
            messages=[{"role": "user", "content":
                "Translate this English medical question into natural Chinese as a Chinese clinician would ask it. "
                f"Output only the Chinese question.\n\n{x['question_en']}"}])
        x["question_zh"] = r.choices[0].message.content.strip()
save()

# ---------- 3. ZH -> EN：C1 (GPT-4o-mini) / C2 (opus-mt) ----------
for x in items:
    if "q_c1" not in x:
        r = client.chat.completions.create(   # 与 run_condition1.py 逐字一致
            model="gpt-4o-mini", max_tokens=200,
            messages=[{"role": "user", "content":
                f"Translate this Chinese medical text to English accurately, preserving all medical terminology:\n\n{x['question_zh']}"}])
        x["q_c1"] = r.choices[0].message.content.strip()
if any("q_c2" not in x for x in items):
    tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    mt = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    for x in items:
        if "q_c2" not in x:   # 与 translate.py 一致
            inp = tok([x["question_zh"]], return_tensors="pt", padding=True, truncation=True, max_length=512)
            x["q_c2"] = tok.decode(mt.generate(**inp)[0], skip_special_tokens=True)
save()

# ---------- 4. 检索 + gold hit@3 ----------
if any("docs_c2" not in x for x in items):
    # 先用 torch 编码，再加载 FAISS（macOS 上两者的 OpenMP 同时活跃会 segfault）
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    EMB = {qk: np.ascontiguousarray(enc.encode([x[qk] for x in items]), dtype="float32") for qk in ["q_c1", "q_c2"]}
    del enc
    print("Loading index and abstracts...")
    faiss.omp_set_num_threads(1)
    index = faiss.read_index("pubmed_index.faiss")
    abstracts = json.load(open("pubmed_abstracts.json", encoding="utf-8"))
    for qk, dk in [("q_c1", "docs_c1"), ("q_c2", "docs_c2")]:
        _, I = index.search(EMB[qk], K)
        for x, row in zip(items, I):
            x[dk] = [abstracts[j]["text"] for j in row]
            x[dk + "_pubids"] = [int(abstracts[j]["pubid"]) for j in row]
    del abstracts
    save()
for c in ["c1", "c2"]:
    hit = np.mean([x["pubid"] in x[f"docs_{c}_pubids"] for x in items])
    print(f"gold hit@{K} {c.upper()}: {hit:.3f}")

# ---------- 5. 回答（A/B/C 映射 yes/no/maybe，沿用主实验单字母格式） ----------
OPTS = {"A": "yes", "B": "no", "C": "maybe"}
def ask(question, docs):
    opts = "\n".join(f"{k}: {v}" for k, v in OPTS.items())
    if docs is None:
        prompt = f"""You are a medical expert. Answer the question.
Only reply with a single letter (A, B, or C).

Question: {question}

Options:
{opts}

Answer:"""
    else:
        context = "\n\n".join(docs)   # 与 run_accuracy.py 一致
        prompt = f"""You are a medical expert. Based on the context below, select the correct answer.
Only reply with a single letter (A, B, or C).

Context:
{context}

Question: {question}

Options:
{opts}

Answer:"""
    r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
                                       max_tokens=5, temperature=0)
    t = r.choices[0].message.content.strip()
    return t[0].upper() if t else "X"

CELLS = {
    "c1q_none":   ("q_c1", None),
    "c1q_gold":   ("q_c1", "gold"),
    "c1q_c1docs": ("q_c1", "docs_c1"),
    "c2q_none":   ("q_c2", None),
    "c2q_gold":   ("q_c2", "gold"),
    "c2q_c2docs": ("q_c2", "docs_c2"),
    "c2q_c1docs": ("q_c2", "docs_c1"),
}
cache.setdefault("preds", {})
for cell, (qk, dk) in CELLS.items():
    if cell in cache["preds"]:
        print(f"[skip] {cell}"); continue
    votes = []
    for x in items:
        docs = None if dk is None else ([x["gold_context"]] if dk == "gold" else x[dk])
        votes.append([ask(x[qk], docs) for _ in range(RUNS)])
    preds = [Counter(v).most_common(1)[0][0] for v in votes]
    cache["preds"][cell] = preds
    save()
    acc = np.mean([OPTS.get(p) == x["answer"] for p, x in zip(preds, items)])
    print(f"=== {cell}: {acc:.3f} ===")

# ---------- 6. 汇总 ----------
correct = {c: [OPTS.get(p) == x["answer"] for p, x in zip(cache["preds"][c], items)] for c in CELLS}
json.dump({"items": items, "correct": correct}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved", OUT)
