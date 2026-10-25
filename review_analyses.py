# review_analyses.py
import json, random
from math import comb

SEED, N_BOOT = 42, 10000

def load(fn):
    return json.load(open(fn, encoding="utf-8"))

def exact_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n) if n else 1.0

def boot(vals, lo_q=0.025, hi_q=0.975):
    rng, n = random.Random(SEED), len(vals)
    bs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return bs[int(lo_q * N_BOOT)], bs[int(hi_q * N_BOOT) - 1]

# ---------- 数据 ----------
main = load("accuracy_results_200_clean.json")
IDS = [r["id"] for r in main]
OK = {c.upper(): {r["id"]: r[f"pred_{c}"] == r["correct_answer"] for r in main}
      for c in ["c1", "c2", "c3", "c4", "c5"]}
c1p = load("accuracy_c1prime_200_clean.json")
OK["C1P"] = {r["id"]: r["pred_c1prime"] == r["correct_answer"] for r in c1p}
for k, rows in load("results_matched_context_200.json").items():
    OK["gpt_" + k] = {r["id"]: r["pred"] == r["correct_answer"] for r in rows}
cl = load("results_claude_2x2_200.json")
for k, rows in cl.items():
    if k != "_meta":
        OK["cl_" + k.replace("claude_", "")] = {r["id"]: r["pred"] == r["correct_answer"] for r in rows}

def v(k, ids=IDS):
    return [int(OK[k][i]) for i in ids]

def acc(k, ids=IDS):
    return sum(v(k, ids)) / len(ids)

def paired(x, y, ids=IDS, margin=None):
    """x - y：差值、95% CI、exact McNemar；如给 margin，再做 TOST（90% CI 落在 ±margin 内即等价）"""
    d = [a - b for a, b in zip(v(x, ids), v(y, ids))]
    b = sum(t == 1 for t in d); c = sum(t == -1 for t in d)
    lo, hi = boot(d)
    s = (f"{x:16s} - {y:16s} {acc(x, ids)*100:5.1f} - {acc(y, ids)*100:5.1f} = "
         f"{sum(d)/len(d)*100:+5.1f}  95% CI [{lo*100:+5.1f}, {hi*100:+5.1f}]  "
         f"b/c {b}/{c}  p = {exact_p(b, c):.4f}")
    if margin is not None:
        lo90, hi90 = boot(d, 0.05, 0.95)
        eq = -margin <= lo90 * 100 and hi90 * 100 <= margin
        s += f"  | TOST ±{margin}: 90% CI [{lo90*100:+.1f}, {hi90*100:+.1f}] -> {'equivalent' if eq else 'not shown'}"
    print(s)

def did(a, b, c, d, label, ids=IDS):
    va, vb, vc, vd = v(a, ids), v(b, ids), v(c, ids), v(d, ids)
    x = [(va[j] - vb[j]) - (vc[j] - vd[j]) for j in range(len(ids))]
    lo, hi = boot(x)
    print(f"{label:58s} DiD = {sum(x)/len(x)*100:+5.1f}  95% CI [{lo*100:+5.1f}, {hi*100:+5.1f}]")

# =============================================================================
print("=== M5 / DA1：MT 题干下，C2 docs 是否比随机文档更有害（coherence 的关键检验）===")
paired("gpt_c2q_random", "C2")
did("gpt_c2q_random", "C2", "gpt_c1q_random", "gpt_c1q_c2docs",
    "interaction vs random: (C2q:rand-C2d) - (C1q:rand-C2d)")
did("gpt_c2q_c1docs", "C2", "C1", "gpt_c1q_c2docs",
    "interaction vs C1 docs (论文现有的 9.5)")

print("\n=== M3：prompt 一致的题干效应（随机文档格子与检索格子用同一 prompt）===")
paired("gpt_c1q_random", "gpt_c2q_random")
paired("C1", "gpt_c2q_c1docs")          # C1 docs 下的题干效应

print("\n=== M4：Shapley 分解（含检索的 2x2：题干 {C1,C2} x 证据 {C1 docs, C2 docs}）===")
for model, (a, b, c, d) in {
    "GPT-4o-mini": ("C1", "gpt_c1q_c2docs", "gpt_c2q_c1docs", "C2"),
    "Claude":      ("cl_c1q_c1docs", "cl_c1q_c2docs", "cl_c2q_c1docs", "cl_c2q_c2docs"),
}.items():
    va, vb, vc, vd = v(a), v(b), v(c), v(d)
    q = [((va[j] - vc[j]) + (vb[j] - vd[j])) / 2 for j in range(len(IDS))]   # 题干的 Shapley 值
    p = [((va[j] - vb[j]) + (vc[j] - vd[j])) / 2 for j in range(len(IDS))]   # 证据的 Shapley 值
    total = (sum(va) - sum(vd)) / len(IDS)
    for name, x in [("query", q), ("passages", p)]:
        lo, hi = boot(x)
        print(f"{model:12s} {name:9s} {sum(x)/len(x)*100:+5.2f}  95% CI [{lo*100:+5.1f}, {hi*100:+5.1f}]"
              f"  ({sum(x)/len(x)/total:.0%} of {total*100:.1f})")

print("\n=== M1 / M2：'没有效果' 的说法 -> 配对差值 CI（关键两条加 TOST ±5）===")
paired("C3", "gpt_c3q_c2docs", margin=5)     # 重新检索
paired("C5", "gpt_c5q_c2docs", margin=5)
paired("C1", "C1P", margin=5)                # C1' ≈ C1
paired("C1", "gpt_c1q_none")                 # 正确翻译下检索的作用（GPT）
paired("cl_c1q_c1docs", "cl_c1q_none")       # Claude
paired("gpt_c1q_random", "gpt_c1q_none")
paired("gpt_c2q_c1docs", "gpt_c2q_none")
paired("gpt_c2q_random", "gpt_c2q_none")

print("\n=== M6：三次运行的一致率（三票完全相同的题目比例）===")
for fn, prefix in [("results_matched_context_200.json", "gpt"), ("results_claude_2x2_200.json", "claude")]:
    for k, rows in load(fn).items():
        if k == "_meta":
            continue
        same = sum(len(set(r["votes"])) == 1 for r in rows) / len(rows)
        print(f"{prefix:6s} {k:20s} {same:.1%}")



print("\n=== R3-P1：中文原题（生成器读原文，检索用译文）===")
for k in ["zhq_none", "zhq_c1docs", "zhq_c2docs", "zhq_random"]:
    print(f"{k:12s} {acc('gpt_' + k)*100:5.1f}")
print()
paired("gpt_zhq_none", "gpt_c1q_none")        # 读原文 vs 读 C1 译文（都不检索）
paired("gpt_zhq_c1docs", "gpt_zhq_none")      # 读原文时检索的作用
paired("gpt_zhq_c1docs", "gpt_zhq_c2docs")    # 读原文时 C2 docs 是否有害（关键）
paired("gpt_zhq_random", "gpt_zhq_c2docs")    # 读原文时 C2 docs vs 随机
paired("gpt_zhq_c2docs", "C2")                # 同样的 C2 docs：读原文 vs 读 MT 译文
did("gpt_c2q_c1docs", "C2", "gpt_zhq_c1docs", "gpt_zhq_c2docs",
    "interaction: C2 docs harm under MT query vs under original query")