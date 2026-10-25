import json
import numpy as np

# Load all data
with open("error_annotations_200.json") as f:
    errors = json.load(f)

with open("accuracy_results_200.json") as f:
    acc_data = json.load(f)

with open("selfcheck_scores_bertscore_200.json") as f:
    bert_data = json.load(f)

# Build per-question metrics
c2_acc_per_q = {r["id"]: int(r["pred_c2"] == r["correct_answer"]) for r in acc_data}
c3_acc_per_q = {r["id"]: int(r["pred_c3"] == r["correct_answer"]) for r in acc_data}
c5_acc_per_q = {r["id"]: int(r["pred_c5"] == r["correct_answer"]) for r in acc_data}
bert_c2_per_q = {s["id"]: s["score_c2"] for s in bert_data}

# Separate questions by error type
t1_ids = set(e["id"] for e in errors if "1" in e["error_types"])
t2_ids = set(e["id"] for e in errors if "2" in e["error_types"])
t3_ids = set(e["id"] for e in errors if "3" in e["error_types"])
only_t1 = t1_ids - t2_ids - t3_ids
only_t2 = t2_ids - t1_ids - t3_ids
only_t3 = t3_ids - t1_ids - t2_ids
all_three = t1_ids & t2_ids & t3_ids

def get_metrics(id_set, label):
    if not id_set:
        print(f"\n{label}: no questions")
        return
    c2 = np.mean([c2_acc_per_q[qid] for qid in id_set if qid in c2_acc_per_q])
    c5 = np.mean([c5_acc_per_q[qid] for qid in id_set if qid in c5_acc_per_q])
    bert = np.mean([bert_c2_per_q[qid] for qid in id_set if qid in bert_c2_per_q])
    print(f"\n{label} (n={len(id_set)}):")
    print(f"  C2 Accuracy:   {c2:.3f}")
    print(f"  C5 Accuracy:   {c5:.3f}  (improvement: {c5-c2:+.3f})")
    print(f"  C2 BERTScore:  {bert:.3f}")

print("=== Impact of Each Error Type on C2 Performance (200 questions) ===")
get_metrics(t1_ids, "T1 Terminology error")
get_metrics(t2_ids, "T2 Syntactic distortion")
get_metrics(t3_ids, "T3 Semantic drift")

print("\n=== Error Combination Analysis ===")
get_metrics(only_t1, "Only T1 (terminology only)")
get_metrics(only_t2, "Only T2 (syntactic only)")
get_metrics(only_t3, "Only T3 (semantic drift only)")
get_metrics(all_three, "All three error types")

print("\n=== C2 vs C3/C5 Improvement by Error Type ===")
for id_set, label in [(t1_ids, "T1"), (t2_ids, "T2"), (t3_ids, "T3")]:
    if not id_set:
        continue
    c2 = np.mean([c2_acc_per_q[qid] for qid in id_set if qid in c2_acc_per_q])
    c3 = np.mean([c3_acc_per_q[qid] for qid in id_set if qid in c3_acc_per_q])
    c5 = np.mean([c5_acc_per_q[qid] for qid in id_set if qid in c5_acc_per_q])
    print(f"{label}: C2={c2:.3f} | C3={c3:.3f} ({c3-c2:+.3f}) | C5={c5:.3f} ({c5-c2:+.3f})")