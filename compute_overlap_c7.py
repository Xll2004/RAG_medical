import json
import pandas as pd

TYPES = ["T1", "T2", "T3"]
COLS = {"T1": "T1_terminology", "T2": "T2_syntactic", "T3": "T3_semantic"}
TYPE_TO_CODE = {"T1": "1", "T2": "2", "T3": "3"}

RETAINED = ["annotator2.xlsx", "annotator3.xlsx"]


def read_xlsx(path):
    df = pd.read_excel(path)
    out = {}
    for _, r in df.iterrows():
        item = int(r["item_no"])
        out[item] = {
            t: (pd.notna(r[COLS[t]]) and str(r[COLS[t]]).strip() != "")
            for t in TYPES
        }
    return out


anns = {p: read_xlsx(p) for p in RETAINED}
for p, a in anns.items():
    dens = {t: sum(1 for v in a.values() if v[t]) for t in TYPES}
    print(f"{p}: {len(a)} 题, " + ", ".join(f"{t} {dens[t]}" for t in TYPES))

with open("human_annotation_answer_key.json") as f:
    key_data = json.load(f)
with open("error_annotations_200.json") as f:
    gpt_data = json.load(f)
with open("accuracy_results_200.json") as f:
    acc_data = json.load(f)

item_to_orig = {e["item_no"]: e["original_id"] for e in key_data}
gpt_by_id = {e["id"]: e for e in gpt_data}
acc_by_id = {e["id"]: e for e in acc_data}


def gpt_labels(orig_id):
    codes = set(gpt_by_id[orig_id].get("error_types", []))
    return {t for t in TYPES if TYPE_TO_CODE[t] in codes}


def analyze(label, ref_fn, item_nos):
    n_valid = n_missed = n_c3_fail = n_both = 0
    both_ids = []

    for item_no in sorted(item_nos):
        orig_id = item_to_orig.get(item_no)
        if orig_id is None or orig_id not in gpt_by_id or orig_id not in acc_by_id:
            continue
        ref = ref_fn(item_no)
        if ref is None:
            continue
        n_valid += 1

        missed = bool(ref - gpt_labels(orig_id))
        rec = acc_by_id[orig_id]
        c3_fail = rec["pred_c3"] != rec["correct_answer"]

        if missed:
            n_missed += 1
        if c3_fail:
            n_c3_fail += 1
        if missed and c3_fail:
            n_both += 1
            both_ids.append(orig_id)

    print(f"\n── 参考标注: {label} ──")
    print(f"有效题数                     : {n_valid}")
    print(f"GPT 漏标 ≥1 个错误类型       : {n_missed}  ({n_missed/n_valid:.1%})")
    print(f"C3 纠正后仍答错              : {n_c3_fail}  ({n_c3_fail/n_valid:.1%})")
    print(f"两者同时发生（重叠）         : {n_both}  ({n_both/n_valid:.1%})")
    if n_missed:
        print(f"漏标题中 C3 也失败的比例     : {n_both/n_missed:.1%}")
    if n_c3_fail:
        print(f"C3 失败题中 GPT 也漏标的比例 : {n_both/n_c3_fail:.1%}")
    print(f"重叠题 id: {both_ids}")


a2, a3 = anns[RETAINED[0]], anns[RETAINED[1]]
common = set(a2) & set(a3)


def lab(a, i):
    return {t for t in TYPES if a[i][t]} if i in a else set()


for p, a in anns.items():
    analyze(p, lambda i, a=a: lab(a, i), a.keys())

analyze(
    "并集（任一标注者标了即算错误）→ circularity 影响上界",
    lambda i: lab(a2, i) | lab(a3, i),
    common,
)
analyze(
    "交集（两位一致才算错误）→ 下界",
    lambda i: lab(a2, i) & lab(a3, i),
    common,
)