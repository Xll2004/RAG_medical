# analyze_pubmedqa_transfer.py -- summarise the PubMedQA transfer experiment (paired bootstrap CI + exact McNemar)
import json
import numpy as np
from math import comb

def mcnemar_exact(b, c):
    k, m = min(b, c), b + c
    return min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0

R = json.load(open("results/results_pubmedqa_transfer_200.json", encoding="utf-8"))
items, C = R["items"], {k: np.array(v, dtype=int) for k, v in R["correct"].items()}
n = len(items)
rng = np.random.default_rng(42)
IDX = rng.integers(0, n, (10000, n))

def comp(a, b, label):
    d = C[a] - C[b]
    bs = d[IDX].mean(1) * 100
    bb, cc = int((d == 1).sum()), int((d == -1).sum())
    p = mcnemar_exact(bb, cc)
    print(f"{label:38s} {C[a].mean()*100:5.1f} - {C[b].mean()*100:5.1f} = {d.mean()*100:+5.1f}  "
          f"95% CI [{np.percentile(bs,2.5):+5.1f}, {np.percentile(bs,97.5):+5.1f}]  b/c {bb}/{cc}  p = {p:.4f}")

print("label dist:", {k: sum(x["answer"] == k for x in items) for k in ["yes", "no", "maybe"]})
for c in ["c1", "c2"]:
    print(f"gold hit@3 {c.upper()}: {np.mean([x['pubid'] in x[f'docs_{c}_pubids'] for x in items]):.3f}")
print("\naccuracy:", {k: round(v.mean() * 100, 1) for k, v in C.items()})

print("\n--- Is retrieval useful (evidence dependence) ---")
comp("c1q_c1docs", "c1q_none", "C1q: C1 docs vs none")
comp("c1q_gold", "c1q_none", "C1q: gold vs none")
comp("c2q_c2docs", "c2q_none", "C2q: C2 docs vs none")
print("\n--- End-to-end gap and decomposition ---")
comp("c1q_c1docs", "c2q_c2docs", "C1 vs C2 (end-to-end)")
comp("c1q_none", "c2q_none", "question effect, no docs")
comp("c1q_gold", "c2q_gold", "question effect, gold docs")
comp("c1q_c1docs", "c2q_c1docs", "question effect, C1 docs")
comp("c2q_c1docs", "c2q_c2docs", "passage effect, C2 question")

print("\n--- Split by whether C2 retrieved the gold abstract ---")
h1 = np.array([x["pubid"] in x["docs_c1_pubids"] for x in items])
h2 = np.array([x["pubid"] in x["docs_c2_pubids"] for x in items])
for name, m in [("C1 hit, C2 hit", h1 & h2), ("C1 hit, C2 miss", h1 & ~h2), ("C1 miss", ~h1)]:
    print(f"{name:16s} n={m.sum():3d}  C1 {C['c1q_c1docs'][m].mean()*100:5.1f}  C2 {C['c2q_c2docs'][m].mean()*100:5.1f}  "
          f"C2q+C1docs {C['c2q_c1docs'][m].mean()*100:5.1f}")
