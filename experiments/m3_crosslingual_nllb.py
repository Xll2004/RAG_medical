# m3_crosslingual_nllb.py -- reviewer comment M3
#  (a) C0: cross-lingual dense retriever; the Chinese question retrieves directly from the English corpus (paraphrase-multilingual-MiniLM-L12-v2 +
#      the existing pubmed_index_mminilm.faiss), bypassing translation entirely; the generator reads the Chinese question
#  (b) C6: medium-quality MT (NLLB-200 distilled 600M), retrieval/generation identical to C2, turning C1 vs C2 into a quality gradient
# Usage: python -u experiments/m3_crosslingual_nllb.py | tee logs/m3.log
from review2_common import *          # Set the OMP environment variables first
import numpy as np
import torch

CACHE = "results/m3_cache.json"
cache = L(CACHE) if os.path.exists(CACHE) else {}
save = lambda: json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- 1. NLLB translation ----------
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

# ---------- 2. Retrieval: encode all queries with torch first, then load FAISS (avoids a macOS OpenMP conflict) ----------
if "docs_c0" not in cache or "docs_c6" not in cache:
    from sentence_transformers import SentenceTransformer
    e_ml = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    E0 = np.ascontiguousarray(e_ml.encode(ZHQ, normalize_embeddings=True), dtype="float32")   # Same as run_c0_mminilm_fixed.py
    del e_ml
    e_en = SentenceTransformer("all-MiniLM-L6-v2")
    E6 = np.ascontiguousarray(e_en.encode(C6Q), dtype="float32")                              # Same as run_condition1.py
    del e_en
    import faiss
    faiss.omp_set_num_threads(1)
    abstracts = L("data/pubmed_abstracts.json")
    for path, E, key in [("data/pubmed_index_mminilm.faiss", E0, "docs_c0"), ("data/pubmed_index.faiss", E6, "docs_c6")]:
        print("Loading", path, flush=True)
        index = faiss.read_index(path)
        _, I = index.search(E, 3)
        cache[key] = [[abstracts[j]["text"] for j in row] for row in I]
        del index
    del abstracts
    save()
C0DOCS, C6DOCS = cache["docs_c0"], cache["docs_c6"]

# ---------- 3. Generation (3 runs + majority vote) ----------
run_cells({
    "zhq_c0docs": (ZHQ, C0DOCS),     # Cross-lingual retrieval + reading the original: full pipeline without translation
    "c6q_c6docs": (C6Q, C6DOCS),     # NLLB end-to-end
    "c6q_c1docs": (C6Q, C1DOCS),     # NLLB question + faithful evidence (for decomposition)
    "zhq_c6docs": (ZHQ, C6DOCS),     # Original question + NLLB evidence (for interaction)
}, "results/results_m3_200.json")

# ---------- 4. GEMBA quality scores for the NLLB translations (same prompt as exp3_qe.py; Claude primary scorer, GPT second scorer) ----------
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
