# analyze_new_experiments.py
# Summarise Exp 1 (evidence attribution), Exp 3 (does translation QE predict failure?) and the memorisation check. Reads JSON only, no API calls.
# Usage: python analysis/analyze_new_experiments.py --provider claude   (use --provider gpt once the GPT runs finish)
import argparse, json, os, random
from math import comb
from scipy.stats import fisher_exact, spearmanr


def roc_auc_score(y, x):
    """AUC = P(score of a failed item > score of a successful item), ties count 0.5 (Mann-Whitney); no sklearn dependency"""
    pos = [s for s, t in zip(x, y) if t]; neg = [s for s, t in zip(x, y) if not t]
    from bisect import bisect_left, bisect_right
    neg.sort()
    tot = sum(bisect_left(neg, p) + 0.5 * (bisect_right(neg, p) - bisect_left(neg, p)) for p in pos)
    return tot / (len(pos) * len(neg))

ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["claude", "gpt"], required=True)
args = ap.parse_args()
P = args.provider
SEED, NB = 42, 10000
L = lambda f: json.load(open(f, encoding="utf-8"))
OUT = {}


def ep(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n) if n else 1.0


def boot_mean(x, lo=0.025, hi=0.975):
    r, n = random.Random(SEED), len(x)
    bs = sorted(sum(x[r.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    return bs[int(lo * NB)], bs[int(hi * NB) - 1]


gold = L("data/gold_set_200_translated.json")
IDS = [q["id"] for q in gold]

# ---------- Per-condition correctness (same definitions as paper_numbers.py) ----------
main = L("results/accuracy_results_200_clean.json")
OK = {c.upper(): {r["id"]: r[f"pred_{c}"] == r["correct_answer"] for r in main} for c in ["c1", "c2", "c3", "c4", "c5"]}
OK["C3B"] = {r["id"]: r["pred_c3b"] == r["correct_answer"] for r in L("results/accuracy_results_c3b_200.json")}
OK["C1P"] = {r["id"]: r["pred_c1prime"] == r["correct_answer"] for r in L("results/accuracy_c1prime_200_clean.json")}
for k, rows in L("results/results_claude_2x2_200.json").items():
    if k != "_meta":
        OK["cl_" + k.replace("claude_", "")] = {r["id"]: r["pred"] == r["correct_answer"] for r in rows}

# =============================================================================
print("=" * 70 + f"\nExp 1: evidence attribution ({P})\n" + "=" * 70)
for suffix in ["", "_neutral"]:
    fn = f"results/exp1_attribution_{P}{suffix}.json"
    if not os.path.exists(fn):
        continue
    d = L(fn)
    print(f"\n[{fn}]")
    print(f"{'cell':12s} {'wrong':>5s} {'cite':>5s} {'verif':>5s} {'%verif|wrong':>12s}   {'right':>5s} {'%verif|right':>12s}  {'evid-backed wrong /200':>22s}")
    res = {}
    for cell, rows in d.items():
        w = [r for r in rows if not r["correct"]]
        rt = [r for r in rows if r["correct"]]
        vw = sum(r["verified"] for r in w)
        vr = sum(r["verified"] for r in rt)
        cw = sum((r["passage"] or 0) > 0 for r in w)
        ebw = {r["id"]: (not r["correct"]) and r["verified"] for r in rows}   # Wrong answers that cite a source
        res[cell] = dict(n_wrong=len(w), cite_wrong=cw, verified_wrong=vw, n_right=len(rt), verified_right=vr,
                         evidence_backed_wrong=sum(ebw.values()))
        print(f"{cell:12s} {len(w):5d} {cw:5d} {vw:5d} {vw/len(w):12.1%}   {len(rt):5d} {vr/len(rt):12.1%}  {sum(ebw.values()):22d}")
        res[cell]["_ebw"] = ebw
    # Key comparison: share of wrong answers that cite a source, C2/C2 (mistranslation-coherent evidence) vs the other cells
    if "c2q_c2docs" in res:
        a = res["c2q_c2docs"]
        for other in res:
            if other == "c2q_c2docs":
                continue
            b = res[other]
            _, p = fisher_exact([[a["verified_wrong"], a["n_wrong"] - a["verified_wrong"]],
                                 [b["verified_wrong"], b["n_wrong"] - b["verified_wrong"]]])
            x, y = a["_ebw"], b["_ebw"]
            bb = sum(x[i] and not y[i] for i in IDS); cc = sum(y[i] and not x[i] for i in IDS)
            print(f"  C2/C2 vs {other}: %verified|wrong Fisher p = {p:.3f};  evidence-backed wrong (paired) b/c {bb}/{cc}, p = {ep(bb, cc):.3f}")
    for v in res.values():
        v.pop("_ebw")
    OUT[f"exp1{suffix}"] = res

# =============================================================================
print("\n" + "=" * 70 + f"\nExp 3: translation-quality (QE) score as a failure predictor ({P} judge)\n" + "=" * 70)
qe = {c: {r["id"]: r["score"] for r in rows} for c, rows in L(f"results/exp3_qe_{P}.json").items()}
for c in qe:
    s = [v for v in qe[c].values() if v is not None]
    print(f"  mean QE {c:4s} = {sum(s)/len(s):5.1f}   (fallback {sum(r.get('fallback', False) for r in L(f'results/exp3_qe_{P}.json')[c])})")
OUT["exp3_mean_qe"] = {c: sum(v for v in qe[c].values() if v is not None) / len(qe[c]) for c in qe}

# Retrieval: translation QE score vs overlap with C1 retrieval (Jaccard)
docs = {"C1": L("results/results_condition1_200_clean.json"), "C1P": L("results/results_c1prime_200_clean.json"),
        "C2": L("results/results_condition2_200.json"), "C3": L("results/results_condition3_200.json"),
        "C3B": L("results/results_condition3b_200.json"), "C4": L("results/results_condition4_200.json"),
        "C5": L("results/results_condition5_200.json")}
D = {c: {r["id"]: set(r["retrieved_docs"]) for r in rows} for c, rows in docs.items()}
jac = lambda a, b: len(a & b) / len(a | b) if a | b else 1.0
NONREF = ["C2", "C3", "C3B", "C4", "C5"]          # Low-quality MT and its corrected versions
pairs = [(qe[c][i], jac(D[c][i], D["C1"][i])) for c in NONREF for i in IDS if qe[c][i] is not None]
rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
print(f"\n  retrieval: Spearman(QE, Jaccard with C1 passages), pooled C2-C5 (n={len(pairs)}): rho = {rho:.3f}, p = {p:.2g}")
rho2, p2 = spearmanr([qe["C2"][i] for i in IDS], [jac(D["C2"][i], D["C1"][i]) for i in IDS])
print(f"  retrieval: same, C2 only (n=200): rho = {rho2:.3f}, p = {p2:.2g}")
OUT["exp3_retrieval"] = dict(pooled_rho=rho, pooled_p=p, pooled_n=len(pairs), c2_rho=rho2, c2_p=p2)

# Answers: among questions C1 gets right, does low QE predict the translated pipeline getting it wrong? (excludes questions the generator can't answer anyway)
def auc_ci(items):
    """items: [(qid, score, failed)]; bootstrap clustered by question"""
    by_q = {}
    for qid, s, f in items:
        by_q.setdefault(qid, []).append((s, f))
    qs = list(by_q)
    y = [f for _, _, f in items]; x = [-s for _, s, _ in items]   # Lower score = more likely to fail
    auc = roc_auc_score(y, x)
    r, bs = random.Random(SEED), []
    for _ in range(2000):
        samp = [t for q in (qs[r.randrange(len(qs))] for _ in qs) for t in by_q[q]]
        yy = [f for _, f in samp]
        if 0 < sum(yy) < len(yy):
            bs.append(roc_auc_score(yy, [-s for s, _ in samp]))
    bs.sort()
    return auc, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]

c1_ok = [i for i in IDS if OK["C1"][i]]
items = [(i, qe[c][i], int(not OK[c][i])) for c in NONREF for i in c1_ok if qe[c][i] is not None]
a = auc_ci(items)
print(f"\n  answers (C1-correct questions, pooled C2-C5, n={len(items)} query-question pairs, "
      f"{sum(f for *_, f in items)} failures): AUC = {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]")
items2 = [(i, qe["C2"][i], int(not OK["C2"][i])) for i in c1_ok]
a2 = auc_ci(items2)
print(f"  answers (C1-correct, C2 only, n={len(items2)}, {sum(f for *_, f in items2)} failures): AUC = {a2[0]:.3f} [{a2[1]:.3f}, {a2[2]:.3f}]")
OUT["exp3_answers"] = dict(pooled_auc=a, pooled_n=len(items), pooled_fail=sum(f for *_, f in items),
                           c2_auc=a2, c2_n=len(items2), c2_fail=sum(f for *_, f in items2))

# A simple "agent self-check" rule: distrust this retrieval if QE is below the threshold
for thr in [50, 70]:
    lo = [(i, c) for c in NONREF for i in c1_ok if qe[c][i] is not None and qe[c][i] < thr]
    hi = [(i, c) for c in NONREF for i in c1_ok if qe[c][i] is not None and qe[c][i] >= thr]
    acc_lo = sum(OK[c][i] for i, c in lo) / len(lo); acc_hi = sum(OK[c][i] for i, c in hi) / len(hi)
    j_lo = sum(jac(D[c][i], D["C1"][i]) for i, c in lo) / len(lo); j_hi = sum(jac(D[c][i], D["C1"][i]) for i, c in hi) / len(hi)
    print(f"  threshold {thr}: below n={len(lo)} acc={acc_lo:.1%} Jaccard={j_lo:.3f} | at/above n={len(hi)} acc={acc_hi:.1%} Jaccard={j_hi:.3f}")
    OUT[f"exp3_thr{thr}"] = dict(n_lo=len(lo), acc_lo=acc_lo, jac_lo=j_lo, n_hi=len(hi), acc_hi=acc_hi, jac_hi=j_hi)

# =============================================================================
print("\n" + "=" * 70 + f"\nMemorization probe ({P})\n" + "=" * 70)
mem = L(f"results/exp_memorization_{P}.json")
from collections import Counter
print("  self-report:", dict(Counter(r["seen"] for r in mem)))
print(f"  option E reproduced exactly: {sum(r['e_exact'] for r in mem)}/{len(mem)}")
flag = {r["id"]: r["e_exact"] or r["seen"] == "yes" for r in mem}
fl = [i for i in IDS if flag.get(i)]; nf = [i for i in IDS if not flag.get(i)]
print(f"  flagged as possibly memorized (exact E or self-report yes): {len(fl)}")
OUT["memorization"] = dict(self_report=dict(Counter(r["seen"] for r in mem)), exact_E=sum(r["e_exact"] for r in mem),
                           n_flagged=len(fl))

def gap(a, b, ids):
    d = [int(OK[a][i]) - int(OK[b][i]) for i in ids]
    lo, hi = boot_mean(d)
    return sum(d) / len(d), lo, hi

for lab, (a, b) in {"GPT-4o-mini C1-C2": ("C1", "C2"), "Claude C1/C1-C2/C2": ("cl_c1q_c1docs", "cl_c2q_c2docs")}.items():
    for nm, ids in [("all", IDS), ("flagged", fl), ("not flagged", nf)]:
        g = gap(a, b, ids)
        print(f"  {lab:20s} {nm:12s} n={len(ids):3d}  acc {sum(OK[a][i] for i in ids)/len(ids):.1%} vs "
              f"{sum(OK[b][i] for i in ids)/len(ids):.1%}  gap = {g[0]*100:+.1f} [{g[1]*100:+.1f}, {g[2]*100:+.1f}]")
        OUT["memorization"][f"{lab}|{nm}"] = dict(n=len(ids), gap=g, acc_a=sum(OK[a][i] for i in ids)/len(ids),
                                                   acc_b=sum(OK[b][i] for i in ids)/len(ids))

json.dump(OUT, open(f"results/new_experiments_out_{P}.json", "w"), indent=1, default=float)
