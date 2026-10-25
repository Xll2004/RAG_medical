# analyze_review2.py —— 汇总第二轮审稿补充实验（M2 / M3 / M4）；只依赖 numpy，缺哪个结果文件就跳过哪部分
import json, os, random
from math import comb
import numpy as np

L = lambda f: json.load(open(f, encoding="utf-8"))
ids = [q["id"] for q in L("gold_set_200_translated.json")]
N = len(ids)
IDX = np.random.default_rng(42).integers(0, N, (10000, N))
C = {}   # cell -> 0/1 向量（按 ids 顺序）

def add(path, prefix=""):
    if not os.path.exists(path):
        return False
    for cell, rows in L(path).items():
        m = {r["id"]: int(r["pred"] == r["correct_answer"]) for r in rows}
        C[prefix + cell] = np.array([m[i] for i in ids])
    return True

A = {r["id"]: r for r in L("accuracy_results_200_clean.json")}
for c in ["c1", "c2", "c3", "c5"]:
    C[c.upper()] = np.array([int(A[i][f"pred_{c}"] == A[i]["correct_answer"]) for i in ids])
add("results_matched_context_200.json")

def mcn(b, c):
    m = b + c
    return min(1.0, 2 * sum(comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m) if m else 1.0

def ci(v):
    bs = v[IDX].mean(1) * 100
    return v.mean() * 100, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def comp(a, b, label=None):
    d = C[a] - C[b]
    m, lo, hi = ci(d)
    bb, cc = int((d == 1).sum()), int((d == -1).sum())
    print(f"  {label or a+' - '+b:34s} {C[a].mean()*100:5.1f} - {C[b].mean()*100:5.1f} = {m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]  b/c {bb}/{cc}  p = {mcn(bb, cc):.4f}")

def did(a, b, c, d, label):
    m, lo, hi = ci((C[a] - C[b]) - (C[c] - C[d]))
    print(f"  {label:34s} DiD = {m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]")

def shapley(q1d1, q2d1, q1d2, q2d2, label):
    q = 0.5 * ((C[q1d1] - C[q2d1]) + (C[q1d2] - C[q2d2]))
    p = 0.5 * ((C[q2d1] - C[q2d2]) + (C[q1d1] - C[q1d2]))
    gap = (C[q1d1] - C[q2d2]).mean() * 100
    for name, v in [("question", q), ("passages", p)]:
        m, lo, hi = ci(v)
        print(f"  Shapley {label} {name:9s} {m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]  ({m/gap:.0%} of {gap:.1f})")

def jac(A_, B_):
    j = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(A_, B_)]
    return np.mean(j), sum(x == 0 for x in j)

R1 = L("results_condition1_200_clean.json"); R2 = L("results_condition2_200.json")
C1DOCS = [r["retrieved_docs"] for r in R1]; C2DOCS = [r["retrieved_docs"] for r in R2]

# ======================= M3 =======================
if add("results_m3_200.json"):
    print("=" * 70 + "\nM3: cross-lingual retrieval (C0) and NLLB (C6)\n" + "=" * 70)
    m3 = L("m3_cache.json")
    for k in ["docs_c0", "docs_c6"]:
        j, dis = jac(C1DOCS, m3[k]); print(f"  Jaccard({k}, C1) = {j:.3f}, disjoint {dis}/{N}")
    j, dis = jac(C2DOCS, m3["docs_c6"]); print(f"  Jaccard(docs_c6, C2) = {j:.3f}")
    qe = {p: L(f"exp3_qe_{p}.json") for p in ["claude", "gpt"]}
    for p in ["claude", "gpt"]:
        s = lambda xs: np.mean([x for x in xs if x is not None])
        print(f"  GEMBA ({p}): C1 {s([r['score'] for r in qe[p]['C1']]):.1f}  C6 {s(m3.get(f'qe_c6_{p}', [np.nan])):.1f}  C2 {s([r['score'] for r in qe[p]['C2']]):.1f}")
    print("  accuracy:", {k: round(C[k].mean() * 100, 1) for k in ["C1", "c6q_c6docs", "C2", "zhq_c0docs", "zhq_c1docs", "zhq_random"]})
    comp("C1", "c6q_c6docs", "C1 vs C6 (NLLB)")
    comp("c6q_c6docs", "C2", "C6 vs C2")
    comp("c1q_c1docs", "c6q_c1docs", "question effect C1->C6 (C1 docs)")
    comp("c6q_c1docs", "c6q_c6docs", "passage effect (C6 question)")
    did("c6q_c1docs", "c6q_c6docs", "zhq_c1docs", "zhq_c6docs", "interaction C6 (vs original reader)")
    comp("zhq_c0docs", "C1", "C0 (no translation) vs C1")
    comp("zhq_c0docs", "zhq_c1docs", "C0 docs vs C1 docs (orig. question)")
    comp("zhq_c0docs", "zhq_random", "C0 docs vs random (orig. question)")

# ======================= M2 =======================
if add("results_m2_medcpt_200.json"):
    print("=" * 70 + "\nM2: MedCPT + MedRAG textbooks\n" + "=" * 70)
    m2 = L("m2_cache.json")
    j, dis = jac(m2["docs_c1m"], m2["docs_c2m"]); print(f"  Jaccard(C1, C2) = {j:.3f}, disjoint {dis}/{N}")
    print("  accuracy:", {k: round(v.mean() * 100, 1) for k, v in C.items() if k.endswith("m")})
    comp("c1q_c1m", "c2q_c2m", "C1 vs C2 (end-to-end)")
    shapley("c1q_c1m", "c2q_c1m", "c1q_c2m", "c2q_c2m", "MedCPT")
    print("  -- does faithful evidence help now? --")
    comp("c1q_c1m", "c1q_randm", "C1q: C1 evidence vs random")
    comp("zhq_c1m", "zhq_randm", "orig q: C1 evidence vs random")
    comp("c1q_c1m", "C1", "MedCPT C1 vs PubMed C1")
    print("  -- mistranslation-coherent evidence --")
    comp("c2q_c1m", "c2q_c2m", "C2q: C1 vs C2 evidence")
    comp("c2q_randm", "c2q_c2m", "C2q: random vs C2 evidence")
    comp("zhq_c1m", "zhq_c2m", "orig q: C1 vs C2 evidence")
    did("c2q_c1m", "c2q_c2m", "zhq_c1m", "zhq_c2m", "interaction vs original reader")
    did("c2q_c1m", "c2q_c2m", "c1q_c1m", "c1q_c2m", "interaction vs C1 reader")
    comp("zhq_c2m", "c2q_c2m", "recovery: orig q + C2 evidence")

# ======================= M4 =======================
if os.path.exists("m4_relevance_gpt.json"):
    print("=" * 70 + "\nM4: closed loop (default OPUS-MT; check fails -> LLM re-translate + re-retrieve = C1 path)\n" + "=" * 70)
    rel = L("m4_relevance_gpt.json")
    ok_rel = {k: np.array([r["verdict"].startswith("yes") for r in rel[k]]) for k in rel}
    print(f"  self-check 'relevant': C2 evidence {ok_rel['c2docs'].sum()}/{N}, C1 evidence {ok_rel['c1docs'].sum()}/{N}")
    gates = {"relevance self-check": ~ok_rel["c2docs"]}
    for p in ["claude", "gpt"]:
        qe = {r["id"]: r["score"] for r in L(f"exp3_qe_{p}.json")["C2"]}
        gates[f"GEMBA<50 ({p})"] = np.array([qe[i] is not None and qe[i] < 50 for i in ids])
    print(f"  never escalate (C2) {C['C2'].mean()*100:.1f} | always escalate (C1) {C['C1'].mean()*100:.1f}")
    for name, g in gates.items():
        acc = np.where(g, C["C1"], C["C2"])
        acc2 = np.where(g, C["zhq_c2docs"], C["C2"])
        rs = []
        for s in range(2000):
            r = np.zeros(N, bool); r[random.Random(s).sample(range(N), int(g.sum()))] = True
            rs.append(np.where(r, C["C1"], C["C2"]).mean() * 100)
        lo, hi = np.percentile(rs, [2.5, 97.5])
        print(f"  {name:22s} escalates {g.sum():3d}/{N}: re-translate {acc.mean()*100:5.1f} | answer original q {acc2.mean()*100:5.1f} | random gate {np.mean(rs):5.1f} [{lo:.1f}, {hi:.1f}]")
