# analyze_matched_context.py
import json, random
from math import comb
from scipy.stats import binomtest

def exact_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n) if n else 1.0

def or_ci(b, c):
    if b + c == 0:
        return float("nan"), float("nan"), float("nan")
    ci = binomtest(b, b + c).proportion_ci(method="exact")
    f = lambda p: p / (1 - p) if p < 1 else float("inf")
    return (b / c if c else float("inf")), f(ci.low), f(ci.high)

def boot_ci(v, n_boot=10000, seed=42):
    rng = random.Random(seed)
    n = len(v)
    accs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot)]

# ---------- Data ----------
main = json.load(open("results/accuracy_results_200_clean.json"))
ids = [r["id"] for r in main]
ok = {c.upper(): {r["id"]: r[f"pred_{c}"] == r["correct_answer"] for r in main}
      for c in ["c1", "c2", "c3", "c5"]}

mc = json.load(open("results/results_matched_context_200.json"))
for cell, rows in mc.items():
    assert [r["id"] for r in rows] == ids, f"{cell}: id order mismatch"
    ok[cell] = {r["id"]: r["pred"] == r["correct_answer"] for r in rows}

# ---------- sanity ----------
if "c1q_c1docs" in mc:
    pred_c1 = {r["id"]: r["pred_c1"] for r in main}
    agree = sum(r["pred"] == pred_c1[r["id"]] for r in mc["c1q_c1docs"])
    print(f"sanity: c1q_c1docs vs clean pred_c1 agreement = {agree}/{len(ids)}\n")

# ---------- Per-cell accuracy + bootstrap 95% CI ----------
print(f"{'cell':12s} {'acc':>6s}  95% CI")
for k, v in ok.items():
    vals = [v[i] for i in ids]
    lo, hi = boot_ci(vals)
    print(f"{k:12s} {sum(vals)/len(vals):6.3f}  [{lo:.3f}, {hi:.3f}]")

# ---------- Secondary-family comparisons ----------
COMPARISONS = [
    ("C1",         "c1q_none",   "Effect of retrieval under faithful translation"),
    ("c1q_c1docs", "c1q_c2docs", "Harm from wrong evidence"),
    ("c1q_c1docs", "c1q_c3docs", "C3 evidence vs C1 evidence"),
    ("c1q_c1docs", "c1q_c5docs", "C5 evidence vs C1 evidence"),
    ("c1q_c2docs", "c1q_random", "Plausible-but-wrong evidence vs random noise"),
    ("c1q_none",   "c1q_random", "Harm from random noise itself"),
    ("c1q_none",   "c2q_none",   "Loss from question translation alone (no retrieval)"),
    ("C2",         "c2q_none",   "Effect of retrieval under MT"),
    ("C3",         "c3q_c2docs", "Re-retrieval (C3)"),
    ("C5", "c5q_c2docs", "Re-retrieval (C5)"),
    ("c2q_none", "c2q_c1docs", "MT question + correct evidence"),
    ("c2q_c1docs", "C2", "MT question: correct vs wrong evidence"),
    ("c2q_none", "c2q_random", "MT question + random noise"),
]
COMPARISONS = [c for c in COMPARISONS if c[0] in ok and c[1] in ok]
alpha = 0.05 / len(COMPARISONS)
print(f"\nsecondary family: m = {len(COMPARISONS)}, Bonferroni alpha = {alpha:.4f}\n")
print(f"{'x':11s} {'y':11s} {'acc x':>6s} {'acc y':>6s} {'b':>4s} {'c':>4s} "
      f"{'exact p':>9s}  OR [95% CI]          question")
for x, y, q in COMPARISONS:
    b = sum(ok[x][i] and not ok[y][i] for i in ids)   # x correct, y wrong
    c = sum(ok[y][i] and not ok[x][i] for i in ids)   # x wrong, y correct
    p = exact_p(b, c)
    o, lo, hi = or_ci(b, c)
    ax = sum(ok[x].values()) / len(ids)
    ay = sum(ok[y].values()) / len(ids)
    star = "*" if p < alpha else " "
    print(f"{x:11s} {y:11s} {ax:6.3f} {ay:6.3f} {b:4d} {c:4d} {p:9.5f}{star} "
          f"{o:5.2f} [{lo:.2f}, {hi:.2f}]   {q}")

# ---------- Interaction: is retrieval more harmful under MT? ----------
# DiD = (C1 - c1q_none) - (C2 - c2q_none), paired bootstrap over questions
v = {k: [int(ok[k][i]) for i in ids] for k in ["C1", "c1q_none", "C2", "c2q_none"]}
d = [(v["C1"][j] - v["c1q_none"][j]) - (v["C2"][j] - v["c2q_none"][j]) for j in range(len(ids))]
rng = random.Random(42)
boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
print(f"\ninteraction DiD = {sum(d)/len(d):+.3f}  95% CI [{boots[250]:+.3f}, {boots[9749]:+.3f}]")

# ---------- Interaction 2: does the harm from C2 docs appear only with the MT question? ----------
# DiD2 = (c2q_c1docs - C2) - (c1q_c1docs - c1q_c2docs)
v2 = {k: [int(ok[k][i]) for i in ids] for k in ["c2q_c1docs", "C2", "c1q_c1docs", "c1q_c2docs"]}
d2 = [(v2["c2q_c1docs"][j] - v2["C2"][j]) - (v2["c1q_c1docs"][j] - v2["c1q_c2docs"][j])
      for j in range(len(ids))]
rng = random.Random(42)
boots2 = sorted(sum(d2[rng.randrange(len(d2))] for _ in d2) / len(d2) for _ in range(10000))
print(f"interaction (doc type x question) DiD = {sum(d2)/len(d2):+.3f}  "
      f"95% CI [{boots2[250]:+.3f}, {boots2[9749]:+.3f}]")