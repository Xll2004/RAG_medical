# analyze_claude_2x2.py
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

def did_boot(d, n_boot=10000, seed=42):
    rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]

# ---------- Data ----------
main = json.load(open("results/accuracy_results_200_clean.json"))
ids = [r["id"] for r in main]
gpt = {"c1q_c1docs": {r["id"]: r["pred_c1"] == r["correct_answer"] for r in main},
       "c2q_c2docs": {r["id"]: r["pred_c2"] == r["correct_answer"] for r in main}}
mc = json.load(open("results/results_matched_context_200.json"))
for k in ["c1q_none", "c2q_none", "c2q_c1docs", "c1q_c2docs"]:
    gpt[k] = {r["id"]: r["pred"] == r["correct_answer"] for r in mc[k]}

cl_raw = json.load(open("results/results_claude_2x2_200.json"))
claude = {}
for k, rows in cl_raw.items():
    if k == "_meta":
        continue
    assert [r["id"] for r in rows] == ids, f"{k}: id order mismatch"
    claude[k.replace("claude_", "")] = {r["id"]: r["pred"] == r["correct_answer"] for r in rows}

acc = lambda d: sum(d[i] for i in ids) / len(ids)

# ---------- Side-by-side comparison ----------
CELLS = [("c1q_none", "C1 question + no retrieval"), ("c1q_c1docs", "C1 question + C1 docs"),
         ("c1q_c2docs", "C1 question + C2 docs"), ("c2q_none", "C2 question + no retrieval"),
         ("c2q_c1docs", "C2 question + C1 docs"), ("c2q_c2docs", "C2 question + C2 docs")]
print(f"{'cell':12s} {'GPT-4o-mini':>12s} {'Claude':>8s}   description")
for k, desc in CELLS:
    g = f"{acc(gpt[k]):.3f}" if k in gpt else "-"
    c = f"{acc(claude[k]):.3f}" if k in claude else "-"
    print(f"{k:12s} {g:>12s} {c:>8s}   {desc}")

print("\nfallback per run:", cl_raw.get("_meta", {}))

# ---------- Claude replication family ----------
COMP = [
    ("c1q_c1docs", "c1q_none",   "Effect of retrieval under faithful translation"),
    ("c1q_none",   "c2q_none",   "Loss from question translation alone (no retrieval)"),
    ("c2q_c2docs", "c2q_none",   "Effect of retrieval under MT"),
    ("c2q_c1docs", "c2q_c2docs", "MT question: correct vs wrong evidence"),
    ("c1q_c1docs", "c1q_c2docs", "C1 question: correct vs wrong evidence"),
    ("c2q_none",   "c2q_c1docs", "MT question + correct evidence"),
]
COMP = [c for c in COMP if c[0] in claude and c[1] in claude]
alpha = 0.05 / len(COMP)
print(f"\nClaude replication family: m = {len(COMP)}, Bonferroni alpha = {alpha:.4f}\n")
print(f"{'x':11s} {'y':11s} {'acc x':>6s} {'acc y':>6s} {'b':>4s} {'c':>4s} "
      f"{'exact p':>9s}  OR [95% CI]          question")
for x, y, q in COMP:
    b = sum(claude[x][i] and not claude[y][i] for i in ids)
    c = sum(claude[y][i] and not claude[x][i] for i in ids)
    p = exact_p(b, c)
    o, lo, hi = or_ci(b, c)
    star = "*" if p < alpha else " "
    print(f"{x:11s} {y:11s} {acc(claude[x]):6.3f} {acc(claude[y]):6.3f} {b:4d} {c:4d} "
          f"{p:9.5f}{star} {o:5.2f} [{lo:.2f}, {hi:.2f}]   {q}")

# ---------- Interaction effect (computed for both models) ----------
def did(m, a, b, c, d):
    v = {k: [int(m[k][i]) for i in ids] for k in [a, b, c, d]}
    return did_boot([(v[a][j] - v[b][j]) - (v[c][j] - v[d][j]) for j in range(len(ids))])

print()
for name, m in [("GPT-4o-mini", gpt), ("Claude", claude)]:
    if all(k in m for k, _ in CELLS):
        e1 = did(m, "c1q_c1docs", "c1q_none", "c2q_c2docs", "c2q_none")
        e2 = did(m, "c2q_c1docs", "c2q_c2docs", "c1q_c1docs", "c1q_c2docs")
        print(f"{name:12s} retrieval x question  DiD = {e1[0]:+.3f}  95% CI [{e1[1]:+.3f}, {e1[2]:+.3f}]")
        print(f"{name:12s} doc type x question   DiD = {e2[0]:+.3f}  95% CI [{e2[1]:+.3f}, {e2[2]:+.3f}]")