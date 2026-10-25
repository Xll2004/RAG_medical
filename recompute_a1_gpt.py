"""
重新计算 Annotator1 vs GPT-4o-mini Jaccard，含逐题明细
运行方式：
  conda activate rag-medical
  cd ~/PycharmProjects/rag_medical_project
  python recompute_a1_gpt.py
"""

import json
import numpy as np

# ── Annotator1 标签（直接从Excel读取，√=True，None=False） ──
raw = {
    1:{"T1":"√","T2":"√","T3":None}, 2:{"T1":"√","T2":None,"T3":None},
    3:{"T1":"√","T2":None,"T3":None}, 4:{"T1":"√","T2":None,"T3":"√"},
    5:{"T1":"√","T2":None,"T3":None}, 6:{"T1":None,"T2":"√","T3":None},
    7:{"T1":"√","T2":None,"T3":None}, 8:{"T1":"√","T2":None,"T3":None},
    9:{"T1":"√","T2":"√","T3":None}, 10:{"T1":None,"T2":None,"T3":None},
    11:{"T1":"√","T2":None,"T3":None}, 12:{"T1":None,"T2":"√","T3":None},
    13:{"T1":"√","T2":None,"T3":"√"}, 14:{"T1":None,"T2":None,"T3":None},
    15:{"T1":None,"T2":"√","T3":"√"}, 16:{"T1":None,"T2":None,"T3":None},
    17:{"T1":"√","T2":"√","T3":None}, 18:{"T1":None,"T2":None,"T3":None},
    19:{"T1":None,"T2":None,"T3":None}, 20:{"T1":"√","T2":None,"T3":"√"},
    21:{"T1":None,"T2":"√","T3":None}, 22:{"T1":None,"T2":"√","T3":None},
    23:{"T1":None,"T2":None,"T3":"√"}, 24:{"T1":None,"T2":None,"T3":None},
    25:{"T1":None,"T2":None,"T3":"√"}, 26:{"T1":None,"T2":None,"T3":"√"},
    27:{"T1":"√","T2":"√","T3":None}, 28:{"T1":"√","T2":None,"T3":None},
    29:{"T1":None,"T2":None,"T3":"√"}, 30:{"T1":"√","T2":None,"T3":None},
    31:{"T1":None,"T2":None,"T3":None}, 32:{"T1":None,"T2":None,"T3":"√"},
    33:{"T1":"√","T2":"√","T3":None}, 34:{"T1":"√","T2":None,"T3":None},
    35:{"T1":"√","T2":"√","T3":None}, 36:{"T1":None,"T2":"√","T3":None},
    37:{"T1":None,"T2":"√","T3":None}, 38:{"T1":None,"T2":None,"T3":None},
    39:{"T1":None,"T2":None,"T3":None}, 40:{"T1":None,"T2":"√","T3":None},
    41:{"T1":"√","T2":None,"T3":None}, 42:{"T1":None,"T2":None,"T3":None},
    43:{"T1":None,"T2":None,"T3":"√"}, 44:{"T1":None,"T2":"√","T3":None},
    45:{"T1":"√","T2":None,"T3":None}, 46:{"T1":"√","T2":None,"T3":None},
    47:{"T1":None,"T2":"√","T3":None}, 48:{"T1":"√","T2":None,"T3":None},
    49:{"T1":"√","T2":"√","T3":None}, 50:{"T1":"√","T2":None,"T3":"√"},
    51:{"T1":None,"T2":None,"T3":None}, 52:{"T1":"√","T2":None,"T3":None},
    53:{"T1":"√","T2":None,"T3":None}, 54:{"T1":"√","T2":None,"T3":None},
    55:{"T1":None,"T2":None,"T3":"√"}, 56:{"T1":None,"T2":"√","T3":None},
    57:{"T1":"√","T2":"√","T3":None}, 58:{"T1":None,"T2":None,"T3":None},
    59:{"T1":None,"T2":None,"T3":None}, 60:{"T1":None,"T2":None,"T3":None},
}
annotator1 = {i: {k: (v == "√") for k, v in d.items()} for i, d in raw.items()}

# ── 加载GPT标签 ──
with open("error_annotations_200.json") as f:
    gpt_data = json.load(f)
with open("human_annotation_answer_key.json") as f:
    key_data = json.load(f)

item_to_orig = {e["item_no"]: e["original_id"] for e in key_data}
gpt_by_id    = {e["id"]: e for e in gpt_data}


def get_gpt(item_no):
    orig_id = item_to_orig[item_no]
    g = gpt_by_id.get(orig_id, {})
    types = g.get("error_types", [])
    return {"T1": "1" in types, "T2": "2" in types, "T3": "3" in types}


gpt_all = {i: get_gpt(i) for i in range(1, 61)}

# ── 逐题明细 ──
print(f"{'item':>4} | {'A1':13} | {'GPT':13} | Jaccard")
print("-" * 50)
scores = []
for i in range(1, 61):
    la = {t for t, v in annotator1[i].items() if v}
    lb = {t for t, v in gpt_all[i].items() if v}
    if len(la | lb) == 0:
        j = 1.0
    else:
        j = len(la & lb) / len(la | lb)
    scores.append(j)
    a1_str  = ",".join(sorted(la)) if la  else "none"
    gpt_str = ",".join(sorted(lb)) if lb  else "none"
    flag = " ←" if j < 0.3 else ""
    print(f"{i:4} | {a1_str:13} | {gpt_str:13} | {j:.3f}{flag}")

print(f"\nAnnotator1 vs GPT (n=60): Mean Jaccard = {np.mean(scores):.3f}")
print(f"  Median: {np.median(scores):.3f}")
print(f"  Perfect agreement (J=1.0): {sum(1 for s in scores if s==1.0)}/60")
print(f"  Zero agreement   (J=0.0): {sum(1 for s in scores if s==0.0)}/60")