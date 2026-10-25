# mcnemar_compare.py
import json
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import binomtest

ALPHA = 0.05 / 11
FILES = {"old": "accuracy_results_200.json", "clean": "accuracy_results_200_clean.json"}
PAIRS = [("c1", "c2"), ("c1", "c3"), ("c1", "c4"), ("c1", "c5"),
         ("c2", "c3"), ("c2", "c4"), ("c2", "c5"),
         ("c3", "c4"), ("c3", "c5"), ("c4", "c5")]

def stats(data, x, y):
    cx = [r[f"pred_{x}"] == r["correct_answer"] for r in data]
    cy = [r[f"pred_{y}"] == r["correct_answer"] for r in data]
    a = sum(p and q for p, q in zip(cx, cy))
    b = sum(p and not q for p, q in zip(cx, cy))      # x 对, y 错
    c = sum(q and not p for p, q in zip(cx, cy))      # x 错, y 对
    d = len(data) - a - b - c
    table = [[a, b], [c, d]]
    p_asym = mcnemar(table, exact=False).pvalue        # 原脚本的做法
    p_exact = mcnemar(table, exact=True).pvalue
    # OR = b/c，95% CI 来自 b/(b+c) 的 Clopper-Pearson 区间
    ci = binomtest(b, b + c).proportion_ci(method="exact")
    or_ = b / c if c else float("inf")
    or_lo = ci.low / (1 - ci.low)
    or_hi = ci.high / (1 - ci.high) if ci.high < 1 else float("inf")
    return b, c, p_asym, p_exact, or_, or_lo, or_hi

data = {k: json.load(open(v)) for k, v in FILES.items()}
for k, d in data.items():
    acc = {c: sum(r[f"pred_{c}"] == r["correct_answer"] for r in d) / len(d)
           for c in ["c1", "c2", "c3", "c4", "c5"]}
    print(f"{k:5s} acc: " + "  ".join(f"{c.upper()}={v:.3f}" for c, v in acc.items()))

print(f"\nBonferroni alpha = {ALPHA:.5f}   (* = significant)\n")
print(f"{'pair':8s} | {'old b/c':>8s} {'asym':>8s} {'exact':>8s} | "
      f"{'clean b/c':>9s} {'asym':>8s} {'exact':>8s} | OR [95% CI] (clean)")
for x, y in PAIRS:
    o = stats(data["old"], x, y)
    n = stats(data["clean"], x, y)
    star = lambda p: f"{p:.4f}{'*' if p < ALPHA else ' '}"
    print(f"{x.upper()}v{y.upper():5s} | {o[0]:>3d}/{o[1]:<4d} {star(o[2])} {star(o[3])} | "
          f"{n[0]:>4d}/{n[1]:<4d} {star(n[2])} {star(n[3])} | "
          f"{n[4]:.2f} [{n[5]:.2f}, {n[6]:.2f}]")