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

# ---------- 数据 ----------
main = json.load(open("accuracy_results_200_clean.json"))
ids = [r["id"] for r in main]
ok = {c.upper(): {r["id"]: r[f"pred_{c}"] == r["correct_answer"] for r in main}
      for c in ["c1", "c2", "c3", "c5"]}

mc = json.load(open("results_matched_context_200.json"))
for cell, rows in mc.items():
    assert [r["id"] for r in rows] == ids, f"{cell}: id 顺序不一致"
    ok[cell] = {r["id"]: r["pred"] == r["correct_answer"] for r in rows}

# ---------- sanity ----------
if "c1q_c1docs" in mc:
    pred_c1 = {r["id"]: r["pred_c1"] for r in main}
    agree = sum(r["pred"] == pred_c1[r["id"]] for r in mc["c1q_c1docs"])
    print(f"sanity: c1q_c1docs vs clean pred_c1 agreement = {agree}/{len(ids)}\n")

# ---------- 各格子准确率 + bootstrap 95% CI ----------
print(f"{'cell':12s} {'acc':>6s}  95% CI")
for k, v in ok.items():
    vals = [v[i] for i in ids]
    lo, hi = boot_ci(vals)
    print(f"{k:12s} {sum(vals)/len(vals):6.3f}  [{lo:.3f}, {hi:.3f}]")

# ---------- secondary family 比较 ----------
COMPARISONS = [
    ("C1",         "c1q_none",   "检索在正确翻译下的作用"),
    ("c1q_c1docs", "c1q_c2docs", "错误证据的伤害"),
    ("c1q_c1docs", "c1q_c3docs", "C3 证据 vs C1 证据"),
    ("c1q_c1docs", "c1q_c5docs", "C5 证据 vs C1 证据"),
    ("c1q_c2docs", "c1q_random", "似是而非证据 vs 随机噪声"),
    ("c1q_none",   "c1q_random", "随机噪声本身的伤害"),
    ("c1q_none",   "c2q_none",   "题干翻译本身的损失（无检索）"),
    ("C2",         "c2q_none",   "MT 条件下检索的作用"),
    ("C3",         "c3q_c2docs", "重新检索 (C3)"),
    ("C5", "c5q_c2docs", "重新检索 (C5)"),
    ("c2q_none", "c2q_c1docs", "MT 题干 + 正确证据"),
    ("c2q_c1docs", "C2", "MT 题干下：正确证据 vs 错误证据"),
    ("c2q_none", "c2q_random", "MT 题干 + 随机噪声"),
]
COMPARISONS = [c for c in COMPARISONS if c[0] in ok and c[1] in ok]
alpha = 0.05 / len(COMPARISONS)
print(f"\nsecondary family: m = {len(COMPARISONS)}, Bonferroni alpha = {alpha:.4f}\n")
print(f"{'x':11s} {'y':11s} {'acc x':>6s} {'acc y':>6s} {'b':>4s} {'c':>4s} "
      f"{'exact p':>9s}  OR [95% CI]          question")
for x, y, q in COMPARISONS:
    b = sum(ok[x][i] and not ok[y][i] for i in ids)   # x 对, y 错
    c = sum(ok[y][i] and not ok[x][i] for i in ids)   # x 错, y 对
    p = exact_p(b, c)
    o, lo, hi = or_ci(b, c)
    ax = sum(ok[x].values()) / len(ids)
    ay = sum(ok[y].values()) / len(ids)
    star = "*" if p < alpha else " "
    print(f"{x:11s} {y:11s} {ax:6.3f} {ay:6.3f} {b:4d} {c:4d} {p:9.5f}{star} "
          f"{o:5.2f} [{lo:.2f}, {hi:.2f}]   {q}")

# ---------- 交互效应：检索的伤害在 MT 下是否更大 ----------
# DiD = (C1 - c1q_none) - (C2 - c2q_none)，按题配对 bootstrap
v = {k: [int(ok[k][i]) for i in ids] for k in ["C1", "c1q_none", "C2", "c2q_none"]}
d = [(v["C1"][j] - v["c1q_none"][j]) - (v["C2"][j] - v["c2q_none"][j]) for j in range(len(ids))]
rng = random.Random(42)
boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
print(f"\ninteraction DiD = {sum(d)/len(d):+.3f}  95% CI [{boots[250]:+.3f}, {boots[9749]:+.3f}]")

# ---------- 交互效应 2：C2 docs 的伤害是否只在 MT 题干下出现 ----------
# DiD2 = (c2q_c1docs - C2) - (c1q_c1docs - c1q_c2docs)
v2 = {k: [int(ok[k][i]) for i in ids] for k in ["c2q_c1docs", "C2", "c1q_c1docs", "c1q_c2docs"]}
d2 = [(v2["c2q_c1docs"][j] - v2["C2"][j]) - (v2["c1q_c1docs"][j] - v2["c1q_c2docs"][j])
      for j in range(len(ids))]
rng = random.Random(42)
boots2 = sorted(sum(d2[rng.randrange(len(d2))] for _ in d2) / len(d2) for _ in range(10000))
print(f"interaction (doc type x question) DiD = {sum(d2)/len(d2):+.3f}  "
      f"95% CI [{boots2[250]:+.3f}, {boots2[9749]:+.3f}]")