import ast
import pandas as pd

TYPES = ["T1", "T2", "T3"]
COLS = {"T1": "T1_terminology", "T2": "T2_syntactic", "T3": "T3_semantic"}


def read_xlsx(path, mark_means_present=True):
    df = pd.read_excel(path)
    out = {}
    for _, r in df.iterrows():
        item = int(r["item_no"])
        row = {}
        for t in TYPES:
            v = r[COLS[t]]
            marked = pd.notna(v) and str(v).strip() != ""
            row[t] = marked if mark_means_present else (not marked)
        out[item] = row
    return out


dicts = {}
for script in ["compute_agreement.py", "compute_agreement_v2.py"]:
    tree = ast.parse(open(script).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if n.startswith("annotator"):
                try:
                    dicts[f"{script}::{n}"] = ast.literal_eval(node.value)
                except Exception:
                    pass


def agree_rate(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0, 0
    hits = sum(
        1 for i in common for t in TYPES if a[i].get(t) == b[i].get(t)
    )
    return hits / (len(common) * 3), len(common)


for path in ["annotator1.xlsx", "annotator2.xlsx"]:
    for polarity in [True, False]:
        xl = read_xlsx(path, polarity)
        tag = "标记=有错误" if polarity else "标记=无错误"
        print(f"\n=== {path} ({tag}) ===")
        for name, d in dicts.items():
            rate, n = agree_rate(xl, d)
            flag = "  <<< 完全匹配" if rate > 0.99 else ""
            print(f"  vs {name:45s} 逐格一致率 {rate:.1%} (n={n}){flag}")


print("\n=== 各来源的标注密度（标记为 True 的比例）===")
for path in ["annotator1.xlsx", "annotator2.xlsx"]:
    xl = read_xlsx(path, True)
    for t in TYPES:
        c = sum(1 for v in xl.values() if v[t])
        print(f"  {path:20s} {t}: {c}/{len(xl)} ({c/len(xl):.1%})")
for name, d in dicts.items():
    for t in TYPES:
        c = sum(1 for v in d.values() if v.get(t))
        print(f"  {name:45s} {t}: {c}/{len(d)} ({c/len(d):.1%})")