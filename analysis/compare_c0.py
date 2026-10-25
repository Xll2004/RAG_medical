# compare_c0.py -- compute C0 results dynamically from JSON, exact McNemar
import json
from scipy.stats import binomtest

def load(fn):
    return {r["id"]: r["correct"] for r in json.load(open(fn))}

acc = json.load(open("results/accuracy_results_200.json"))
c1 = {r["id"]: int(r["pred_c1"] == r["correct_answer"]) for r in acc}

conds = {
    "C0_old (mismatched index)": load("results/results_c0_multilingual_200.json"),
    "C0_fixed":                  load("results/results_c0_mminilm_fixed_200.json"),
    "C0_noRAG_zh":               load("results/results_c0_norag_zh_200.json"),
    "C1":                        c1,
}
ids = sorted(c1)
for name, d in conds.items():
    assert set(d) == set(ids), f"{name}: id set mismatch"
    print(f"{name:28s} acc = {sum(d[i] for i in ids)/len(ids)*100:.1f}%")

def mcnemar(a, b, la, lb):
    b_only = sum(1 for i in ids if a[i] == 1 and b[i] == 0)
    c_only = sum(1 for i in ids if a[i] == 0 and b[i] == 1)
    p = binomtest(b_only, b_only + c_only, 0.5).pvalue if b_only + c_only else 1.0
    print(f"{la} vs {lb}: {la}-only={b_only}, {lb}-only={c_only}, exact p={p:.4f}")

print()
mcnemar(conds["C0_fixed"], conds["C0_noRAG_zh"], "C0_fixed", "C0_noRAG_zh")   # Main comparison: does multilingual retrieval help?
mcnemar(conds["C0_fixed"], conds["C0_old (mismatched index)"], "C0_fixed", "C0_old")
mcnemar(conds["C0_fixed"], conds["C1"], "C0_fixed", "C1")                      # Reference only; protocols are not equivalent
