"""
计算标注一致性：
- Annotator1 vs GPT-4o-mini (60题)
- Annotator2 vs GPT-4o-mini (21题，基于文字反馈整理的标签)
- Annotator1 vs Annotator2 (21题重叠部分)

运行方式：
  conda activate rag-medical
  python compute_agreement.py
"""

import json
import numpy as np
from sklearn.metrics import cohen_kappa_score

# ── 加载数据 ──────────────────────────────────────────────────
with open("human_annotation_answer_key.json") as f:
    key_data = json.load(f)

with open("error_annotations_200.json") as f:
    gpt_data = json.load(f)

# 建立映射
item_to_orig = {e["item_no"]: e["original_id"] for e in key_data}
gpt_by_id    = {e["id"]: e for e in gpt_data}

# ── Annotator 1 标签（从Excel读取，手动整理） ─────────────────
# 格式：{item_no: {"T1": bool, "T2": bool, "T3": bool}}
annotator1 = {
    1:  {"T1": True,  "T2": True,  "T3": False},
    2:  {"T1": True,  "T2": False, "T3": False},
    3:  {"T1": True,  "T2": False, "T3": False},
    4:  {"T1": True,  "T2": False, "T3": True},
    5:  {"T1": True,  "T2": False, "T3": False},
    6:  {"T1": False, "T2": True,  "T3": False},
    7:  {"T1": True,  "T2": False, "T3": False},
    8:  {"T1": True,  "T2": False, "T3": False},
    9:  {"T1": True,  "T2": True,  "T3": False},
    10: {"T1": False, "T2": False, "T3": False},
    11: {"T1": True,  "T2": False, "T3": False},
    12: {"T1": False, "T2": True,  "T3": False},
    13: {"T1": True,  "T2": False, "T3": True},
    14: {"T1": False, "T2": False, "T3": False},
    15: {"T1": False, "T2": True,  "T3": True},
    16: {"T1": False, "T2": False, "T3": False},
    17: {"T1": True,  "T2": True,  "T3": False},
    18: {"T1": False, "T2": False, "T3": False},
    19: {"T1": False, "T2": False, "T3": False},
    20: {"T1": True,  "T2": False, "T3": True},
    21: {"T1": False, "T2": True,  "T3": False},
    22: {"T1": False, "T2": True,  "T3": False},
    23: {"T1": False, "T2": False, "T3": True},
    24: {"T1": False, "T2": False, "T3": False},
    25: {"T1": False, "T2": False, "T3": True},
    26: {"T1": False, "T2": False, "T3": True},
    27: {"T1": True,  "T2": True,  "T3": False},
    28: {"T1": True,  "T2": False, "T3": False},
    29: {"T1": False, "T2": False, "T3": True},
    30: {"T1": True,  "T2": False, "T3": False},
    31: {"T1": False, "T2": False, "T3": False},
    32: {"T1": False, "T2": False, "T3": True},
    33: {"T1": True,  "T2": True,  "T3": False},
    34: {"T1": True,  "T2": False, "T3": False},
    35: {"T1": True,  "T2": True,  "T3": False},
    36: {"T1": False, "T2": True,  "T3": False},
    37: {"T1": False, "T2": True,  "T3": False},
    38: {"T1": False, "T2": False, "T3": False},
    39: {"T1": False, "T2": False, "T3": False},
    40: {"T1": False, "T2": True,  "T3": False},
    41: {"T1": True,  "T2": False, "T3": False},
    42: {"T1": False, "T2": False, "T3": False},
    43: {"T1": False, "T2": False, "T3": True},
    44: {"T1": False, "T2": True,  "T3": False},
    45: {"T1": True,  "T2": False, "T3": False},
    46: {"T1": True,  "T2": False, "T3": False},
    47: {"T1": False, "T2": True,  "T3": False},
    48: {"T1": True,  "T2": False, "T3": False},
    49: {"T1": True,  "T2": True,  "T3": False},
    50: {"T1": True,  "T2": False, "T3": True},
    51: {"T1": False, "T2": False, "T3": False},
    52: {"T1": True,  "T2": False, "T3": False},
    53: {"T1": True,  "T2": False, "T3": False},
    54: {"T1": True,  "T2": False, "T3": False},
    55: {"T1": False, "T2": False, "T3": True},
    56: {"T1": False, "T2": True,  "T3": False},
    57: {"T1": True,  "T2": True,  "T3": False},
    58: {"T1": False, "T2": False, "T3": False},
    59: {"T1": False, "T2": False, "T3": False},
    60: {"T1": False, "T2": False, "T3": False},
}

# ── Annotator 2 标签（21题文字反馈整理） ─────────────────────
# item_no对应关系：Q1=item1, Q3=item3, Q5=item5...以此类推
annotator2_items = {
    1:  {"T1": True,  "T2": False, "T3": True},   # "eye and blood" = TCM五轮，T1+T3
    3:  {"T1": True,  "T2": True,  "T3": False},   # "high menstrual"+"illustrated"，T1+T2
    5:  {"T1": True,  "T2": True,  "T3": True},    # "each other's roots"，三类全有
    6:  {"T1": True,  "T2": False, "T3": True},    # "I don't know what I'm talking about"，T1+T3
    8:  {"T1": True,  "T2": False, "T3": False},   # "incorrigible drug"，T1
    9:  {"T1": True,  "T2": True,  "T3": False},   # "vertical low blood pressure"，T1+T2
    10: {"T1": True,  "T2": False, "T3": False},   # "applicable"，T1
    11: {"T1": True,  "T2": True,  "T3": False},   # "acetic acetic sermons"，T1+T2
    12: {"T1": True,  "T2": True,  "T3": True},    # "suspended 45 days"+"vagina"，三类全有
    13: {"T1": True,  "T2": False, "T3": True},    # "obsessiveness of the sun"，T1+T3
    17: {"T1": True,  "T2": False, "T3": False},   # "weak echo"，T1
    20: {"T1": True,  "T2": True,  "T3": False},   # "drain more urinately"，T1+T2
    34: {"T1": True,  "T2": False, "T3": True},    # "when you're jealous"="醋制"，T1+T3
    35: {"T1": True,  "T2": True,  "T3": False},   # "impregnable room velocity"，T1+T2
    36: {"T1": True,  "T2": False, "T3": False},   # "increase of 2 years"，T1
    46: {"T1": True,  "T2": False, "T3": True},    # "mammothing"，T1+T3
    47: {"T1": False, "T2": False, "T3": True},    # "liver and blood go missing"，T3
    52: {"T1": False, "T2": False, "T3": False},   # 翻译基本正确，无错误
    54: {"T1": True,  "T2": False, "T3": True},    # "hydro-flying"，T1+T3
    55: {"T1": True,  "T2": False, "T3": False},   # "target-adaptation patients"，T1
    57: {"T1": True,  "T2": False, "T3": False},   # "medically ordered not to keep"，T1
}


def get_gpt_labels(item_no):
    orig_id = item_to_orig[item_no]
    g = gpt_by_id.get(orig_id, {})
    types = g.get("error_types", [])
    return {
        "T1": "1" in types,
        "T2": "2" in types,
        "T3": "3" in types,
    }


def jaccard(a, b):
    """Per-question Jaccard between two label sets."""
    scores = []
    for item_no in a:
        if item_no not in b:
            continue
        la = {t for t, v in a[item_no].items() if v}
        lb = {t for t, v in b[item_no].items() if v}
        if len(la | lb) == 0:
            scores.append(1.0)  # both empty = perfect agreement
        else:
            scores.append(len(la & lb) / len(la | lb))
    return np.mean(scores), len(scores)


def cohen_kappa_per_type(a, b, shared_items):
    """Cohen's kappa per error type, over shared items."""
    results = {}
    for t in ["T1", "T2", "T3"]:
        y1 = [int(a[i][t]) for i in shared_items if i in a and i in b]
        y2 = [int(b[i][t]) for i in shared_items if i in a and i in b]
        if len(set(y1)) < 2 and len(set(y2)) < 2:
            results[t] = "N/A (no variance)"
        else:
            results[t] = f"{cohen_kappa_score(y1, y2):.3f}"
    return results


# ── GPT labels for all items ──────────────────────────────────
gpt_all    = {i: get_gpt_labels(i) for i in range(1, 61)}
gpt_21     = {i: get_gpt_labels(i) for i in annotator2_items}

# ── 1. Annotator1 vs GPT (60题) ───────────────────────────────
j1_gpt, n1 = jaccard(annotator1, gpt_all)
kappa1_gpt  = cohen_kappa_per_type(annotator1, gpt_all, list(range(1, 61)))

print("=" * 55)
print(f"Annotator1 vs GPT-4o-mini  (n={n1})")
print(f"  Mean Jaccard: {j1_gpt:.3f}")
print(f"  Cohen's kappa per type: {kappa1_gpt}")

# ── 2. Annotator2 vs GPT (21题) ───────────────────────────────
j2_gpt, n2 = jaccard(annotator2_items, gpt_21)
kappa2_gpt  = cohen_kappa_per_type(annotator2_items, gpt_21,
                                   list(annotator2_items.keys()))

print()
print(f"Annotator2 vs GPT-4o-mini  (n={n2})")
print(f"  Mean Jaccard: {j2_gpt:.3f}")
print(f"  Cohen's kappa per type: {kappa2_gpt}")

# ── 3. Annotator1 vs Annotator2 (21题重叠) ───────────────────
shared = list(annotator2_items.keys())
a1_21   = {i: annotator1[i] for i in shared}
j12, n12 = jaccard(a1_21, annotator2_items)
kappa12  = cohen_kappa_per_type(a1_21, annotator2_items, shared)

print()
print(f"Annotator1 vs Annotator2   (n={n12})")
print(f"  Mean Jaccard: {j12:.3f}")
print(f"  Cohen's kappa per type: {kappa12}")

print()
print("=" * 55)
print("Summary for paper:")
print(f"  Human inter-annotator Jaccard (n=21): {j12:.3f}")
print(f"  Annotator1 vs GPT Jaccard (n=60):     {j1_gpt:.3f}")
print(f"  Annotator2 vs GPT Jaccard (n=21):     {j2_gpt:.3f}")
