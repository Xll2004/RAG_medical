import ast
import glob
import os
import pandas as pd

# ── 定位两份 Excel ──
paths = {}
for name in ["annotator1.xlsx", "annotator2.xlsx"]:
    hits = glob.glob(f"./**/{name}", recursive=True)
    hits += glob.glob(os.path.expanduser(f"~/Downloads/{name}"))
    hits += glob.glob(os.path.expanduser(f"~/Desktop/{name}"))
    if hits:
        paths[name] = hits[0]
    print(f"{name}: {hits if hits else '未找到'}")

# ── 解析 v1 / v2 里硬编码的 dict ──
dicts = {}
for script in ["compute_agreement.py", "compute_agreement_v2.py"]:
    if not os.path.exists(script):
        continue
    tree = ast.parse(open(script).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if n.startswith("annotator"):
                try:
                    dicts[f"{script}::{n}"] = ast.literal_eval(node.value)
                except Exception:
                    pass

print("\n── 脚本里的 dict ──")
for k, v in dicts.items():
    print(f"{k}: {len(v)} 题")

# ── 读 Excel，打印结构 ──
sheets = {}
for name, p in paths.items():
    df = pd.read_excel(p)
    sheets[name] = df
    print(f"\n── {name} ({p}) ──")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(5).to_string())