# paper_numbers.py
"""
Single source of every number in the paper. Reads JSON only, no API calls.
Conventions:
  - C1 / C1' use the clean versions (meta-text removed)
  - The generator always sees the original Chinese options (main-experiment protocol)
  - All McNemar tests are exact (two-sided binomial), reporting b/c, OR and 95% CI
  - Matched-context cells for "C1 question + C1 docs" and "C2 question + C2 docs" use the main-experiment C1 / C2;
    c1q_c1docs is used only as a reproducibility check
"""
import ast, json, random, re
from collections import Counter
from itertools import combinations
from math import comb

import pandas as pd
from scipy.stats import binomtest

SEED, N_BOOT = 42, 10000
MAIN_M = 11            # Number of tests in the main family (paper Table II: pairwise C1-C5 + C3 vs C3b)
OUT = {}               # All numbers; written to paper_numbers_out.json at the end

def load(fn):
    return json.load(open(fn, encoding="utf-8"))

def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)

# ---------- Statistics helpers ----------
def exact_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n) if n else 1.0

def or_ci(b, c):
    if b + c == 0:
        return float("nan"), float("nan"), float("nan")
    ci = binomtest(b, b + c).proportion_ci(method="exact")
    f = lambda p: p / (1 - p) if p < 1 else float("inf")
    return (b / c if c else float("inf")), f(ci.low), f(ci.high)

def boot_ci(vals):
    rng, n = random.Random(SEED), len(vals)
    accs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return accs[int(0.025 * N_BOOT)], accs[int(0.975 * N_BOOT) - 1]

# ---------- Data ----------
gold = {q["id"]: q for q in load("data/gold_set_200_translated.json")}
main = load("results/accuracy_results_200_clean.json")
IDS = [r["id"] for r in main]
assert set(IDS) == set(gold), "gold ids do not match the main results"

OK = {}   # condition name -> {id: correct?}
for c in ["c1", "c2", "c3", "c4", "c5"]:
    OK[c.upper()] = {r["id"]: r[f"pred_{c}"] == r["correct_answer"] for r in main}

def ok_from(fn, label):
    rows = load(fn)
    key = next(k for k in rows[0] if k.startswith("pred") and not k.endswith("_raw"))
    print(f"  {label}: {fn} [{key}]")
    return {r["id"]: r[key] == r["correct_answer"] for r in rows}

print("Loading:")
OK["C3B"] = ok_from("results/accuracy_results_c3b_200.json", "C3b")
OK["C1P"] = ok_from("results/accuracy_c1prime_200_clean.json", "C1'")

mc = load("results/results_matched_context_200.json")
for k, rows in mc.items():
    OK["gpt_" + k] = {r["id"]: r["pred"] == r["correct_answer"] for r in rows}

cl = load("results/results_claude_2x2_200.json")
for k, rows in cl.items():
    if k != "_meta":
        OK["cl_" + k.replace("claude_", "")] = {r["id"]: r["pred"] == r["correct_answer"] for r in rows}

for k, v in OK.items():
    assert set(v) == set(IDS), f"{k}: id mismatch"

def acc(k, ids=IDS):
    return sum(OK[k][i] for i in ids) / len(ids)

def compare(x, y, ids=IDS):
    b = sum(OK[x][i] and not OK[y][i] for i in ids)   # x correct, y wrong
    c = sum(OK[y][i] and not OK[x][i] for i in ids)   # x wrong, y correct
    o, lo, hi = or_ci(b, c)
    return dict(acc_x=acc(x, ids), acc_y=acc(y, ids), b=b, c=c, p=exact_p(b, c),
                OR=o, OR_lo=lo, OR_hi=hi)

def print_family(name, pairs, m=None, ids=IDS):
    m = m or len(pairs)
    alpha = 0.05 / m
    print(f"\n[{name}]  m = {m}, Bonferroni alpha = {alpha:.5f}   (* = significant)")
    print(f"{'x':16s} {'y':16s} {'acc x':>6s} {'acc y':>6s} {'diff':>6s} {'b':>4s} {'c':>4s} "
          f"{'exact p':>9s}  OR [95% CI]          note")
    rows = []
    for x, y, note in pairs:
        r = compare(x, y, ids)
        star = "*" if r["p"] < alpha else " "
        print(f"{x:16s} {y:16s} {r['acc_x']:6.3f} {r['acc_y']:6.3f} "
              f"{(r['acc_x']-r['acc_y'])*100:+6.1f} {r['b']:4d} {r['c']:4d} {r['p']:9.5f}{star} "
              f"{r['OR']:5.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]   {note}")
        rows.append(dict(x=x, y=y, note=note, significant=r["p"] < alpha, **r))
    OUT.setdefault("families", {})[name] = dict(m=m, alpha=alpha, tests=rows)

def did(a, b, c, d, ids=IDS):
    """DiD = (a - b) - (c - d), paired bootstrap over questions"""
    v = {k: [int(OK[k][i]) for i in ids] for k in [a, b, c, d]}
    diffs = [(v[a][j] - v[b][j]) - (v[c][j] - v[d][j]) for j in range(len(ids))]
    rng = random.Random(SEED)
    bs = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs)
                for _ in range(N_BOOT))
    return sum(diffs) / len(diffs), bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT) - 1]

# =============================================================================
section("1. Accuracy with bootstrap 95% CI")
print(f"{'condition':18s} {'acc':>6s}  95% CI")
OUT["accuracy"] = {}
for k in OK:
    lo, hi = boot_ci([int(OK[k][i]) for i in IDS])
    OUT["accuracy"][k] = dict(acc=acc(k), ci_lo=lo, ci_hi=hi)
    print(f"{k:18s} {acc(k):6.3f}  [{lo:.3f}, {hi:.3f}]")

# =============================================================================
section("2. Main test family (paper Table II: pairwise C1-C5 + C3 vs C3b)")
print_family("main", [
    ("C1", "C2", ""), ("C1", "C3", ""), ("C1", "C4", ""), ("C1", "C5", ""),
    ("C2", "C3", ""), ("C2", "C4", ""), ("C2", "C5", ""),
    ("C3", "C4", ""), ("C3", "C5", ""), ("C4", "C5", ""),
    ("C3", "C3B", ""),
], m=MAIN_M)

# =============================================================================
section("3. Secondary family: GPT-4o-mini factorial (matched context)")
print_family("secondary_gpt", [
    ("C1", "gpt_c1q_none",               "Effect of retrieval under faithful translation"),
    ("C1", "gpt_c1q_c2docs",             "C1 question: C1 docs vs C2 docs"),
    ("C1", "gpt_c1q_c3docs",             "C1 question: C1 docs vs C3 docs"),
    ("C1", "gpt_c1q_c5docs",             "C1 question: C1 docs vs C5 docs"),
    ("gpt_c1q_c2docs", "gpt_c1q_random", "C1 question: C2 docs vs random"),
    ("gpt_c1q_none", "gpt_c1q_random",   "C1 question: no retrieval vs random"),
    ("gpt_c1q_none", "gpt_c2q_none",     "Loss from question translation (no retrieval)"),
    ("C2", "gpt_c2q_none",               "Effect of retrieval with the MT question"),
    ("gpt_c2q_c1docs", "C2",             "MT question: C1 docs vs C2 docs"),
    ("gpt_c2q_none", "gpt_c2q_c1docs",   "MT question: no retrieval vs C1 docs"),
    ("gpt_c2q_none", "gpt_c2q_random",   "MT question: no retrieval vs random"),
    ("C3", "gpt_c3q_c2docs",             "Effect of re-retrieval (C3)"),
    ("C5", "gpt_c5q_c2docs",             "Effect of re-retrieval (C5)"),
    ("C1", "C1P",                        "C1 vs C1'"),
    ("C1P", "gpt_c1q_none",              "C1' vs no retrieval"),
])

# =============================================================================
section("4. Claude replication family")
print_family("replication_claude", [
    ("cl_c1q_c1docs", "cl_c1q_none",     "Effect of retrieval under faithful translation"),
    ("cl_c1q_none", "cl_c2q_none",       "Loss from question translation (no retrieval)"),
    ("cl_c2q_c2docs", "cl_c2q_none",     "Effect of retrieval with the MT question"),
    ("cl_c2q_c1docs", "cl_c2q_c2docs",   "MT question: C1 docs vs C2 docs"),
    ("cl_c1q_c1docs", "cl_c1q_c2docs",   "C1 question: C1 docs vs C2 docs"),
    ("cl_c2q_none", "cl_c2q_c1docs",     "MT question: no retrieval vs C1 docs"),
    ("cl_c1q_c1docs", "C1",              "Claude vs GPT (C1 condition, replaces the original C1b)"),
])

# =============================================================================
section("5. 2x2 tables and interaction effects")
GRID = {
    "GPT-4o-mini": dict(q1_none="gpt_c1q_none", q1_d1="C1", q1_d2="gpt_c1q_c2docs",
                        q2_none="gpt_c2q_none", q2_d1="gpt_c2q_c1docs", q2_d2="C2"),
    "Claude":      dict(q1_none="cl_c1q_none", q1_d1="cl_c1q_c1docs", q1_d2="cl_c1q_c2docs",
                        q2_none="cl_c2q_none", q2_d1="cl_c2q_c1docs", q2_d2="cl_c2q_c2docs"),
}
OUT["grid"] = {}
for model, g in GRID.items():
    print(f"\n{model}")
    print(f"  {'':10s} {'no docs':>8s} {'C1 docs':>8s} {'C2 docs':>8s}")
    print(f"  {'C1 query':10s} {acc(g['q1_none']):8.3f} {acc(g['q1_d1']):8.3f} {acc(g['q1_d2']):8.3f}")
    print(f"  {'C2 query':10s} {acc(g['q2_none']):8.3f} {acc(g['q2_d1']):8.3f} {acc(g['q2_d2']):8.3f}")
    e1 = did(g["q1_d1"], g["q1_none"], g["q2_d2"], g["q2_none"])     # retrieval x question
    e2 = did(g["q2_d1"], g["q2_d2"], g["q1_d1"], g["q1_d2"])         # document type x question
    print(f"  retrieval x query  DiD = {e1[0]*100:+.1f}  95% CI [{e1[1]*100:+.1f}, {e1[2]*100:+.1f}]")
    print(f"  doc type  x query  DiD = {e2[0]*100:+.1f}  95% CI [{e2[1]*100:+.1f}, {e2[2]*100:+.1f}]")
    OUT["grid"][model] = dict(cells={k: acc(v) for k, v in g.items()},
                              did_retrieval_x_query=e1, did_doctype_x_query=e2)

# =============================================================================
section("6. Output format: invalid and fallback")
inv_main = {c: sum(r[f"pred_{c}"] not in list("ABCDE") for r in main) for c in ["c1","c2","c3","c4","c5"]}
inv_mc = {k: sum(x not in list("ABCDE") for r in v for x in r["votes"]) for k, v in mc.items()}
inv_cl = {k: sum(x not in list("ABCDE") for r in v for x in r["votes"]) for k, v in cl.items() if k != "_meta"}
fb_cl = {k: sum(v["fallback_per_run"]) for k, v in cl.get("_meta", {}).items()}
tot_mc = sum(len(v) * 3 for v in mc.values())
tot_cl = sum(len(v) * 3 for k, v in cl.items() if k != "_meta")
print(f"GPT main: non-letter final predictions per condition: {inv_main}")
print(f"GPT matched-context: invalid votes {sum(inv_mc.values())}/{tot_mc} "
      f"({sum(inv_mc.values())/tot_mc*100:.2f}%)  {inv_mc}")
print(f"Claude: invalid votes {sum(inv_cl.values())}/{tot_cl} ({sum(inv_cl.values())/tot_cl*100:.2f}%)")
print(f"Claude: fallback {sum(fb_cl.values())}/{tot_cl} ({sum(fb_cl.values())/tot_cl*100:.2f}%)  {fb_cl}")
OUT["format"] = dict(gpt_main=inv_main, gpt_mc=inv_mc, claude_invalid=inv_cl, claude_fallback=fb_cl)

# =============================================================================
section("7. Retrieval overlap (Jaccard): C1 (clean) vs other conditions")
docs = {"C1": load("results/results_condition1_200_clean.json"),
        "C1_raw": load("results/results_condition1_200.json")}
for c in [2, 3, 4, 5]:
    docs[f"C{c}"] = load(f"results/results_condition{c}_200.json")
D = {k: {r["id"]: set(r["retrieved_docs"]) for r in v} for k, v in docs.items()}

def jac(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0

# First reproduce the original retrieval_overlap_200.json with raw C1, to confirm the Jaccard definition matches
old = load("results/retrieval_overlap_200.json")
for key, rows in old.items():
    if not isinstance(rows, list):
        print(f"  retrieval_overlap_200.json['{key}'] (summary) = {json.dumps(rows, ensure_ascii=False)[:300]}")
        continue
    match = [c for c in ["C2", "C3", "C4", "C5"]
             if all(abs(jac(D["C1_raw"][r["id"]], D[c][r["id"]]) - r["jaccard"]) < 1e-9 for r in rows)]
    print(f"  retrieval_overlap_200.json['{key}'] reproduced as C1_raw vs {match or '??? (not reproduced)'}")

OUT["jaccard"] = {}
print(f"\n{'pair':10s} {'mean J':>7s} {'mean ov':>8s}   overlap 0/1/2/3 (questions)   accuracy of that condition at each overlap level")
for c in ["C2", "C3", "C4", "C5"]:
    js = [jac(D["C1"][i], D[c][i]) for i in IDS]
    ov = {i: len(D["C1"][i] & D[c][i]) for i in IDS}
    cnt = [sum(ov[i] == k for i in IDS) for k in range(4)]
    acc_by = []
    for k in range(4):
        sub = [i for i in IDS if ov[i] == k]
        acc_by.append(f"{sum(OK[c][i] for i in sub)/len(sub):.3f}" if sub else "  -  ")
    mean_ov = sum(ov.values()) / len(IDS)
    print(f"C1 vs {c:4s} {sum(js)/len(js):7.3f} {mean_ov:8.3f}   {cnt}   {acc_by}")
    OUT["jaccard"][c] = dict(mean=sum(js)/len(js), mean_overlap=mean_ov,
                             overlap_counts=cnt, acc_by_overlap=acc_by)

# =============================================================================
section("8. Robustness: excluding TCM, regulation and non-medical questions")
KW = {
    # Chinese keywords are matched against the original Chinese subject/question text -- do not translate
    "TCM":         ["中医", "中药", "针灸", "推拿", "中西医", "方剂", "伤寒", "金匮", "温病", "内经"],
    "regulation":  ["法规", "卫生法", "药事管理", "医学伦理", "法律"],
    "non-medical": ["政治", "马克思", "毛泽东", "思想", "英语", "形势与政策"],
}
def text_of(q):
    return " ".join(str(q.get(k, "")) for k in ["exam_type", "exam_class", "exam_subject"])

CAT = {r["id"]: r["category"] for r in load("results/question_classifications_200.json")}
print("  question categories:", dict(sorted(Counter(CAT.values()).items())))
excluded = {}
for i in IDS:
    t = text_of(gold[i])
    if CAT[i] == "REGULATION":           # Regulation questions: follow the paper's existing question-type classification
        excluded[i] = "regulation"
        continue
    for cat, words in KW.items():
        if any(w in t for w in words):
            excluded[i] = cat
            break
for cat in KW:
    subjects = sorted({gold[i]["exam_subject"] for i, c in excluded.items() if c == cat})
    print(f"  {cat:12s} {sum(c == cat for c in excluded.values()):3d} questions   subjects: {subjects}")
KEEP = [i for i in IDS if i not in excluded]
print(f"  Kept {len(KEEP)}/{len(IDS)} questions")

OUT["robustness"] = dict(n_keep=len(KEEP), excluded={str(k): v for k, v in excluded.items()})
print(f"\n  Subset accuracy: " + "  ".join(f"{k}={acc(k, KEEP):.3f}" for k in ["C1", "C2", "C3", "C4", "C5"]))
print_family("robustness_subset", [
    ("C1", "C2", "main result"), ("C1", "C5", "main result"), ("C2", "C5", "main result"),
    ("gpt_c1q_none", "gpt_c2q_none", "Loss from question translation (no retrieval)"),
    ("C1", "gpt_c1q_none", "Effect of retrieval under faithful translation"),
    ("gpt_c2q_c1docs", "C2", "MT question: C1 docs vs C2 docs"),
    ("C3", "gpt_c3q_c2docs", "Re-retrieval (C3)"),
    ("C5", "gpt_c5q_c2docs", "Re-retrieval (C5)"),
    ("cl_c1q_none", "cl_c2q_none", "Claude: loss from question translation"),
    ("cl_c2q_c1docs", "cl_c2q_c2docs", "Claude: MT question, C1 vs C2 docs"),
], ids=KEEP)
for model, g in GRID.items():
    e2 = did(g["q2_d1"], g["q2_d2"], g["q1_d1"], g["q1_d2"], ids=KEEP)
    print(f"  {model:12s} doc type x query DiD (subset) = {e2[0]*100:+.1f}  "
          f"95% CI [{e2[1]*100:+.1f}, {e2[2]*100:+.1f}]")
    OUT["robustness"][f"did_doctype_{model}"] = e2

# =============================================================================
section("9. Reproducibility checks")
pred_c1 = {r["id"]: r["pred_c1"] for r in main}
agree = sum(r["pred"] == pred_c1[r["id"]] for r in mc["c1q_c1docs"])
print(f"c1q_c1docs (rerun) vs main C1: {agree}/{len(IDS)} predictions identical")
OUT["reproducibility_c1"] = agree

# =============================================================================
section("10. Subgroups by question type (all accuracies recomputed on clean data)")
COLS = ["C1", "C2", "C3", "C5", "gpt_c1q_none", "gpt_c2q_none", "cl_c1q_none", "cl_c2q_none"]
print(f"{'category':20s} {'n':>4s} " + " ".join(f"{c:>13s}" for c in COLS) + "   C1-C2 gap")
OUT["subgroup"] = {}
for cat in sorted(set(CAT.values())):
    sub = [i for i in IDS if CAT[i] == cat]
    row = {c: acc(c, sub) for c in COLS}
    gap = (row["C1"] - row["C2"]) * 100
    print(f"{cat:20s} {len(sub):4d} " + " ".join(f"{row[c]:13.3f}" for c in COLS) + f"   {gap:+.1f}")
    OUT["subgroup"][cat] = dict(n=len(sub), gap_c1_c2=gap, **row)

# =============================================================================
section("11. Annotation agreement (60-question human subset)")
MARK = {"annotator1": "√", "annotator2": "X", "annotator3": "√"}   # Marker meaning "has this error type" in each file
COLMAP = {"T1": "T1_terminology", "T2": "T2_syntactic", "T3": "T3_semantic"}
TYPES = ["T1", "T2", "T3", "T23"]          # T23 = T2 or T3 (merged category)

def read_xlsx(name):
    df = pd.read_excel(f"data/{name}.xlsx")
    for c in COLMAP.values():                # Apart from the marker and empty values, no other values are allowed
        vals = set(df[c].dropna().astype(str).str.strip())
        assert vals <= {MARK[name]}, f"{name}.{c} has unexpected values: {vals}"
    return {int(r["item_no"]): {t: str(r[c]).strip() == MARK[name] for t, c in COLMAP.items()}
            for _, r in df.iterrows()}

def load_dicts(path):
    """Read the annotatorN = {...} dicts verbatim from the script (without executing it), for the consistency check"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and re.fullmatch(r"annotator\d+", node.targets[0].id)):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out

def lab(src, item, t):
    return (src[item]["T2"] or src[item]["T3"]) if t == "T23" else src[item][t]

def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

def kripp_alpha(units):
    """Nominal-scale Krippendorff's alpha; units = list of each rater's label per question"""
    o = Counter()
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        cnt = Counter(vals)
        for c in cnt:
            for k in cnt:
                o[(c, k)] += cnt[c] * (cnt[k] - (c == k)) / (m - 1)
    n_c = Counter()
    for (c, k), v in o.items():
        n_c[c] += v
    n = sum(n_c.values())
    dis_obs = sum(v for (c, k), v in o.items() if c != k)
    dis_exp = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k)
    return 1 - (n - 1) * dis_obs / dis_exp if dis_exp else float("nan")

# --- Load ---
key = load("data/human_annotation_answer_key.json")
items = sorted(e["item_no"] for e in key)
SRC = {n: read_xlsx(n) for n in MARK}
SRC["GPT"] = {e["item_no"]: {t: e[f"gpt4o_mini_{t}"] for t in ["T1", "T2", "T3"]} for e in key}
for n, d in SRC.items():
    assert set(d) == set(items), f"{n}: questions do not match the answer key"

# --- Consistency check: Excel vs the dicts in compute_agreement_v2.py ---
for n, d in load_dicts("annotation/compute_agreement_v2.py").items():
    diff = [(i, t) for i in items for t in ["T1", "T2", "T3"] if d[i][t] != SRC[n][i][t]]
    print(f"  {n}: xlsx vs compute_agreement_v2.py dict -> "
          f"{'identical' if not diff else f'{len(diff)} mismatches, e.g. {diff[:5]}'}")

# --- Annotation rates ---
rate = lambda n, t: sum(lab(SRC[n], i, t) for i in items) / len(items)
print(f"\n  {'rater':12s} " + " ".join(f"{t:>6s}" for t in TYPES))
for n in SRC:
    print(f"  {n:12s} " + " ".join(f"{rate(n, t):6.3f}" for t in TYPES))

# --- Pairwise kappa (including GPT's directional bias) ---
OUT["annotation"] = {"pairwise": {}, "alpha": {}}
print(f"\n  {'pair':26s} {'type':5s} {'kappa':>6s} {'over':>5s} {'under':>6s}   "
      f"(over/under only meaningful for GPT: over = GPT marked it but the human did not)")
for a, b in combinations(SRC, 2):
    if a == "GPT":
        a, b = b, a
    for t in TYPES:
        x = [lab(SRC[a], i, t) for i in items]
        y = [lab(SRC[b], i, t) for i in items]
        k = kappa(x, y)
        rec = dict(kappa=k)
        extra = ""
        if b == "GPT":
            rec["over"] = sum(g and not h for g, h in zip(y, x))
            rec["under"] = sum(h and not g for g, h in zip(y, x))
            extra = f" {rec['over']:5d} {rec['under']:6d}"
        print(f"  {a + ' vs ' + b:26s} {t:5s} {k:6.3f}{extra}")
        OUT["annotation"]["pairwise"][f"{a}_vs_{b}_{t}"] = rec

# --- Krippendorff's alpha: all combinations (sensitivity) ---
humans = list(MARK)
SETS = {"all 3 humans": humans}
for a, b in combinations(humans, 2):
    SETS[f"{a[-1]}+{b[-1]}"] = [a, b]
    SETS[f"{a[-1]}+{b[-1]}+GPT"] = [a, b, "GPT"]
SETS["all 3 humans+GPT"] = humans + ["GPT"]
print(f"\n  {'Krippendorff alpha':20s} " + " ".join(f"{t:>6s}" for t in TYPES))
for label, raters in SETS.items():
    row = [kripp_alpha([[lab(SRC[r], i, t) for r in raters] for i in items]) for t in TYPES]
    for t, v in zip(TYPES, row):
        OUT["annotation"]["alpha"][f"{label}_{t}"] = v
    print(f"  {label:20s} " + " ".join(f"{v:6.3f}" for v in row))

# --- Reproduce the paper's Jaccard (mean of per-question label-set Jaccard) ---
def jacc(a, b):
    vals = []
    for i in items:
        A = {t for t in ["T1", "T2", "T3"] if SRC[a][i][t]}
        B = {t for t in ["T1", "T2", "T3"] if SRC[b][i][t]}
        U = A | B
        vals.append(len(A & B) / len(U) if U else 1.0)
    return sum(vals) / len(vals)

print(f"\n  Jaccard (paper values: 2-3 = 0.492; 1-2 / 1-3 = 0.403 / 0.400; "
      f"GPT vs 2 / 3 = 0.594 / 0.678; GPT vs 1 = 0.333)")
OUT["annotation"]["jaccard"] = {}
for a, b in combinations(SRC, 2):
    j = jacc(a, b)
    OUT["annotation"]["jaccard"][f"{a}_vs_{b}"] = j
    print(f"  {a + ' vs ' + b:26s} {j:.3f}")

# =============================================================================
section("12. Derived numbers in the main text")
n = len(IDS)
both    = sum(OK["C1"][i] and OK["C2"][i] for i in IDS)
c1only  = sum(OK["C1"][i] and not OK["C2"][i] for i in IDS)
c2only  = sum(OK["C2"][i] and not OK["C1"][i] for i in IDS)
neither = n - both - c1only - c2only
print(f"C1 vs C2 per-question breakdown: both correct {both} | only C1 correct {c1only} ({c1only/n:.1%}) | "
      f"only C2 correct {c2only} | both wrong {neither} ({neither/n:.1%})")

cnt = OUT["jaccard"]["C2"]["overlap_counts"]
print(f"C1 vs C2 retrieval: no overlap {cnt[0]} ({cnt[0]/n:.1%}) | identical {cnt[3]} ({cnt[3]/n:.1%})")

r = compare("gpt_c1q_none", "C2")
print(f"no-RAG vs C2: {r['acc_x']:.3f} vs {r['acc_y']:.3f}, b/c = {r['b']}/{r['c']}, exact p = {r['p']:.6f}")
print(f"gaps: C1-C2 {(acc('C1')-acc('C2'))*100:.1f} | C1-C5 {(acc('C1')-acc('C5'))*100:.1f} | "
      f"no-RAG - C2 {(acc('gpt_c1q_none')-acc('C2'))*100:.1f}")
OUT["derived"] = dict(both=both, c1_only=c1only, c2_only=c2only, neither=neither,
                      disjoint=cnt[0], identical=cnt[3], norag_vs_c2=r)

# =============================================================================
json.dump(OUT, open("results/paper_numbers_out.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print("\nsaved paper_numbers_out.json")