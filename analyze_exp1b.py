# analyze_exp1b.py —— 汇总新版 Exp 1（生成时的解释是否依据检索段落）
# 用法：python analyze_exp1b.py --provider claude   （有第二个 judge 时再加 --second gpt 算一致性）
import argparse, json
from collections import Counter
from scipy.stats import fisher_exact

ap = argparse.ArgumentParser()
ap.add_argument("--provider", default="claude")
ap.add_argument("--second", default=None)
args = ap.parse_args()
L = lambda f: json.load(open(f, encoding="utf-8"))
J = L(f"exp1b_grounding_{args.provider}.json")
PRED = {c.upper(): {r["id"]: r[f"pred_{c}"] for r in L("accuracy_results_200_clean.json")} for c in ["c1", "c2", "c3", "c4", "c5"]}
PRED["C3B"] = {r["id"]: r["pred_c3b"] for r in L("accuracy_results_c3b_200.json")}

OUT = {}
print(f"{'cond':4s} {'kept':>4s} | {'wrong':>5s} {'evid':>10s} {'claims':>10s} | {'right':>5s} {'evid':>10s} {'claims':>10s}")
for c, rows in J.items():
    kept = [r for r in rows if not r["excluded"] and r["chosen_option"] == PRED[c][r["id"]]]
    res = {"n_total": len(rows), "n_excluded": sum(r["excluded"] for r in rows), "n_kept": len(kept)}
    line = f"{c:4s} {len(kept):4d} |"
    for lab, sub in [("wrong", [r for r in kept if r["chosen_option"] != r["correct_answer"]]),
                     ("right", [r for r in kept if r["chosen_option"] == r["correct_answer"]])]:
        ev = sum(r["span_verified"] for r in sub); cl = sum(r["claims_support"] for r in sub)
        res[lab] = dict(n=len(sub), evidence=ev, claims=cl, grounding=dict(Counter(r["grounding"] for r in sub)),
                        claims_without_evidence=sum(r["claims_support"] and not r["span_verified"] for r in sub))
        line += f" {len(sub):5d} {ev:3d} ({ev/len(sub):4.0%}) {cl:3d} ({cl/len(sub):4.0%}) |"
    print(line)
    OUT[c] = res

def fisher(a, b, key, lab="wrong"):
    x, y = OUT[a][lab], OUT[b][lab]
    _, p = fisher_exact([[x[key], x["n"] - x[key]], [y[key], y["n"] - y[key]]])
    return p

print()
for key in ["evidence", "claims"]:
    for other in ["C3", "C3B", "C4", "C5"]:
        pass
    print(f"  wrong answers, {key}: C2 vs C1 Fisher p = {fisher('C2', 'C1', key):.3f};"
          f"  wrong vs right within C2 p = {fisher_exact([[OUT['C2']['wrong'][key], OUT['C2']['wrong']['n']-OUT['C2']['wrong'][key]],[OUT['C2']['right'][key], OUT['C2']['right']['n']-OUT['C2']['right'][key]]])[1]:.3f}")
# 合并：低质量翻译（C2-C5）vs 忠实翻译（C1）
agg = lambda cs, lab, key: (sum(OUT[c][lab][key] for c in cs), sum(OUT[c][lab]["n"] for c in cs))
for key in ["evidence", "claims", "claims_without_evidence"]:
    a, na = agg(["C2", "C3", "C3B", "C4", "C5"], "wrong", key); b, nb = agg(["C1"], "wrong", key)
    _, p = fisher_exact([[a, na - a], [b, nb - b]])
    print(f"  wrong, {key}: C2-C5 pooled {a}/{na} ({a/na:.1%}) vs C1 {b}/{nb} ({b/nb:.1%}), Fisher p = {p:.3f}")
    OUT[f"pooled_{key}"] = dict(lowq=(a, na), c1=(b, nb), p=p)

if args.second:
    J2 = L(f"exp1b_grounding_{args.second}.json")
    for key in ["grounding", "claims_support", "chosen_option"]:
        agree = n = 0
        for c in J:
            m = {r["id"]: r for r in J2.get(c, [])}
            for r in J[c]:
                if r["id"] in m:
                    n += 1; agree += r[key] == m[r["id"]][key]
        print(f"  judge agreement on {key}: {agree}/{n} = {agree/n:.1%}")
json.dump(OUT, open(f"exp1b_out_{args.provider}.json", "w"), indent=1)
