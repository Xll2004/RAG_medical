import json
import numpy as np
from sklearn.metrics import cohen_kappa_score

with open("human_annotation_answer_key.json") as f:
    key_data = json.load(f)
with open("error_annotations_200.json") as f:
    gpt_data = json.load(f)

item_to_orig = {e["item_no"]: e["original_id"] for e in key_data}
gpt_by_id    = {e["id"]: e for e in gpt_data}

# Annotator 1 (60 questions, Excel)
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

# Annotator 3 (60 questions, structured table)
annotator3 = {
    1:  {"T1": True,  "T2": True,  "T3": True},
    2:  {"T1": True,  "T2": True,  "T3": True},
    3:  {"T1": True,  "T2": True,  "T3": True},
    4:  {"T1": True,  "T2": False, "T3": True},
    5:  {"T1": True,  "T2": False, "T3": True},
    6:  {"T1": True,  "T2": False, "T3": True},
    7:  {"T1": False, "T2": True,  "T3": True},
    8:  {"T1": True,  "T2": False, "T3": True},
    9:  {"T1": True,  "T2": False, "T3": True},
    10: {"T1": True,  "T2": False, "T3": True},
    11: {"T1": True,  "T2": False, "T3": True},
    12: {"T1": True,  "T2": True,  "T3": True},
    13: {"T1": True,  "T2": False, "T3": True},
    14: {"T1": True,  "T2": False, "T3": True},
    15: {"T1": True,  "T2": False, "T3": True},
    16: {"T1": True,  "T2": False, "T3": True},
    17: {"T1": True,  "T2": False, "T3": True},
    18: {"T1": False, "T2": False, "T3": False},
    19: {"T1": False, "T2": False, "T3": False},
    20: {"T1": True,  "T2": True,  "T3": True},
    21: {"T1": True,  "T2": False, "T3": True},
    22: {"T1": True,  "T2": True,  "T3": True},
    23: {"T1": True,  "T2": False, "T3": True},
    24: {"T1": True,  "T2": False, "T3": True},
    25: {"T1": True,  "T2": False, "T3": True},
    26: {"T1": False, "T2": False, "T3": True},
    27: {"T1": True,  "T2": False, "T3": True},
    28: {"T1": True,  "T2": False, "T3": True},
    29: {"T1": True,  "T2": False, "T3": True},
    30: {"T1": True,  "T2": True,  "T3": True},
    31: {"T1": False, "T2": False, "T3": True},
    32: {"T1": True,  "T2": False, "T3": True},
    33: {"T1": True,  "T2": True,  "T3": True},
    34: {"T1": True,  "T2": False, "T3": True},
    35: {"T1": True,  "T2": True,  "T3": True},
    36: {"T1": True,  "T2": False, "T3": True},
    37: {"T1": True,  "T2": False, "T3": True},
    38: {"T1": True,  "T2": False, "T3": False},
    39: {"T1": True,  "T2": False, "T3": False},
    40: {"T1": True,  "T2": True,  "T3": True},
    41: {"T1": True,  "T2": False, "T3": True},
    42: {"T1": True,  "T2": False, "T3": False},
    43: {"T1": True,  "T2": False, "T3": False},
    44: {"T1": False, "T2": False, "T3": False},
    45: {"T1": True,  "T2": False, "T3": True},
    46: {"T1": True,  "T2": False, "T3": False},
    47: {"T1": True,  "T2": False, "T3": False},
    48: {"T1": True,  "T2": False, "T3": False},
    49: {"T1": True,  "T2": True,  "T3": True},
    50: {"T1": True,  "T2": False, "T3": False},
    51: {"T1": True,  "T2": False, "T3": True},
    52: {"T1": True,  "T2": False, "T3": False},
    53: {"T1": True,  "T2": False, "T3": False},
    54: {"T1": True,  "T2": False, "T3": False},
    55: {"T1": False, "T2": False, "T3": False},
    56: {"T1": True,  "T2": False, "T3": True},
    57: {"T1": True,  "T2": False, "T3": True},
    58: {"T1": False, "T2": False, "T3": False},
    59: {"T1": True,  "T2": False, "T3": False},
    60: {"T1": True,  "T2": False, "T3": False},
}


def get_gpt_labels(item_no):
    orig_id = item_to_orig[item_no]
    g = gpt_by_id.get(orig_id, {})
    types = g.get("error_types", [])
    return {"T1": "1" in types, "T2": "2" in types, "T3": "3" in types}


def jaccard(a, b):
    scores = []
    for i in a:
        if i not in b:
            continue
        la = {t for t, v in a[i].items() if v}
        lb = {t for t, v in b[i].items() if v}
        if len(la | lb) == 0:
            scores.append(1.0)
        else:
            scores.append(len(la & lb) / len(la | lb))
    return np.mean(scores), len(scores)


def kappa_per_type(a, b, items):
    results = {}
    for t in ["T1", "T2", "T3"]:
        y1 = [int(a[i][t]) for i in items if i in a and i in b]
        y2 = [int(b[i][t]) for i in items if i in a and i in b]
        if len(set(y1)) < 2 and len(set(y2)) < 2:
            results[t] = "N/A"
        else:
            results[t] = round(cohen_kappa_score(y1, y2), 3)
    return results


gpt_all = {i: get_gpt_labels(i) for i in range(1, 61)}
all_items = list(range(1, 61))

# 1. Annotator1 vs Annotator3 (60 questions)
j13, n13 = jaccard(annotator1, annotator3)
k13 = kappa_per_type(annotator1, annotator3, all_items)

# 2. Annotator1 vs GPT (60 questions)
j1g, n1g = jaccard(annotator1, gpt_all)
k1g = kappa_per_type(annotator1, gpt_all, all_items)

# 3. Annotator3 vs GPT (60 questions)
j3g, n3g = jaccard(annotator3, gpt_all)
k3g = kappa_per_type(annotator3, gpt_all, all_items)

print("=" * 60)
print(f"Annotator1 vs Annotator3 (n={n13})")
print(f"  Mean Jaccard: {j13:.3f}")
print(f"  Kappa per type: {k13}")

print()
print(f"Annotator1 vs GPT-4o-mini (n={n1g})")
print(f"  Mean Jaccard: {j1g:.3f}")
print(f"  Kappa per type: {k1g}")

print()
print(f"Annotator3 vs GPT-4o-mini (n={n3g})")
print(f"  Mean Jaccard: {j3g:.3f}")
print(f"  Kappa per type: {k3g}")

print()
print("=" * 60)
print("KEY NUMBERS FOR PAPER:")
print(f"  Human inter-annotator Jaccard (n=60): {j13:.3f}")
print(f"  Annotator1 vs GPT Jaccard (n=60):     {j1g:.3f}")
print(f"  Annotator3 vs GPT Jaccard (n=60):     {j3g:.3f}")