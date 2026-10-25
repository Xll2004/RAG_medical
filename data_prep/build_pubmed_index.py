"""
Rebuild the PubMedQA retrieval corpus and its FAISS index.

Reconstructed from the paper's description (Sec. III-A); the original build script was not kept.
  - Corpus: every context section of every abstract in the PubMedQA source corpus
    (pqa_labeled + pqa_unlabeled + pqa_artificial), one passage per context section,
    stored as {"pubid", "question", "text"}.
  - Encoder: all-MiniLM-L6-v2 (sentence-transformers defaults, no normalisation).
  - Index: FAISS IndexFlatL2, rows in the same order as the passages.

Outputs go to *_rebuilt files; the originals in data/ are never overwritten.
Each step compares its output with the existing file and reports the first difference.

How to run (from the repository root):
  python data_prep/build_pubmed_index.py                 # corpus only (a few minutes)
  python data_prep/build_pubmed_index.py --index         # corpus + embeddings + FAISS index (slow)
  python data_prep/build_pubmed_index.py --order labeled,artificial,unlabeled   # try another subset order
"""
import argparse
import json
import os

import numpy as np

CORPUS = "data/pubmed_abstracts.json"
INDEX = "data/pubmed_index.faiss"
CORPUS_OUT = "data/pubmed_abstracts_rebuilt.json"
INDEX_OUT = "data/pubmed_index_rebuilt.faiss"

ap = argparse.ArgumentParser()
ap.add_argument("--order", default="labeled,unlabeled,artificial",
                help="order in which the PubMedQA subsets are concatenated")
ap.add_argument("--index", action="store_true", help="also encode the corpus and build the FAISS index")
ap.add_argument("--batch-size", type=int, default=64)
args = ap.parse_args()


# ---------- 1. Corpus ----------
def build_corpus(order):
    from datasets import load_dataset
    passages, seen_pubids, seen_texts = [], set(), set()
    stats = {}
    for name in order:
        ds = load_dataset("qiaojin/PubMedQA", f"pqa_{name}", split="train")
        added_abs = skipped_abs = 0
        for row in ds:
            pubid = int(row["pubid"])
            if pubid in seen_pubids:          # same abstract already taken from an earlier subset
                skipped_abs += 1
                continue
            new = []
            for c in row["context"]["contexts"]:   # skip texts seen anywhere earlier, including in this abstract
                if c not in seen_texts and c not in new:
                    new.append(c)
            if not new:                       # every passage of this abstract is a duplicate
                skipped_abs += 1
                continue
            seen_pubids.add(pubid)
            for c in new:
                seen_texts.add(c)
                passages.append({"pubid": pubid, "question": row["question"], "text": c})
            added_abs += 1
        stats[name] = (len(ds), added_abs, skipped_abs)
    for name, (n, a, s) in stats.items():
        print(f"  pqa_{name:10s} abstracts {n:7d}  kept {a:7d}  skipped {s:4d}")
    return passages


def compare_corpus(new, path):
    if not os.path.exists(path):
        print(f"  {path} not found, nothing to compare against")
        return False
    with open(path, encoding="utf-8") as f:
        old = json.load(f)
    print(f"  rebuilt {len(new)} passages / {len({p['pubid'] for p in new})} abstracts;"
          f"  existing {len(old)} / {len({p['pubid'] for p in old})}")
    for i, (a, b) in enumerate(zip(new, old)):
        if a != b:
            print(f"  FIRST DIFFERENCE at passage {i}:")
            print(f"    rebuilt : pubid={a['pubid']}  {a['text'][:80]!r}")
            print(f"    existing: pubid={b['pubid']}  {b['text'][:80]!r}")
            return False
    if len(new) != len(old):
        print("  identical prefix, but lengths differ")
        return False
    print("  IDENTICAL to the existing corpus")
    return True


print("Building corpus, order:", args.order)
passages = build_corpus(args.order.split(","))
with open(CORPUS_OUT, "w", encoding="utf-8") as f:
    json.dump(passages, f, ensure_ascii=False, indent=2)
print(f"  wrote {CORPUS_OUT}")
same_corpus = compare_corpus(passages, CORPUS)


# ---------- 2. Embeddings + FAISS index ----------
if args.index:
    if not same_corpus:
        print("\nCorpus differs from the existing one, so the index cannot be compared row by row. "
              "Fix the corpus first (try another --order).")
    from sentence_transformers import SentenceTransformer
    import faiss   # import after torch is loaded (macOS OpenMP conflict)

    print("\nEncoding", len(passages), "passages with all-MiniLM-L6-v2 ...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    # Encode in chunks and free the MPS cache between them: padding of varying lengths
    # otherwise makes the MPS cache grow until it runs out of memory (same fix as m2_medcpt.py)
    import torch
    texts = [p["text"] for p in passages]
    CHUNK = 20000
    parts = []
    for start in range(0, len(texts), CHUNK):
        parts.append(encoder.encode(texts[start:start + CHUNK], batch_size=args.batch_size,
                                    convert_to_numpy=True, show_progress_bar=False))
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"  encoded {min(start + CHUNK, len(texts))}/{len(texts)}", flush=True)
    emb = np.ascontiguousarray(np.concatenate(parts), dtype="float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, INDEX_OUT)
    print(f"  wrote {INDEX_OUT}  (ntotal={index.ntotal}, dim={index.d})")

    if same_corpus and os.path.exists(INDEX):
        old = faiss.read_index(INDEX)
        print(f"  existing index: ntotal={old.ntotal}, dim={old.d}")
        rng = np.random.default_rng(42)
        rows = rng.choice(old.ntotal, size=min(2000, old.ntotal), replace=False)
        diff = max(float(np.abs(old.reconstruct(int(r)) - index.reconstruct(int(r))).max()) for r in rows)
        # Embeddings are not bit-identical across hardware/library versions; small differences are expected
        print(f"  max |difference| over {len(rows)} sampled vectors: {diff:.2e}  "
              f"({'MATCH' if diff < 1e-3 else 'DIFFERENT'} at tolerance 1e-3)")
