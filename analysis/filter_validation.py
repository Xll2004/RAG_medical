import json
import pandas as pd

# Load all data
with open("results/accuracy_results_200.json") as f:
    acc_data = json.load(f)
with open("results/error_annotations_200.json") as f:
    errors = json.load(f)
with open("results/option_quality_analysis_200.json") as f:
    option_quality = json.load(f)

# Build per-question data
err_by_id = {e["id"]: e for e in errors}
opt_by_id = {o["id"]: o for o in option_quality}

rows = []
for r in acc_data:
    qid = r["id"]
    err = err_by_id.get(qid, {})
    opt = opt_by_id.get(qid, {})

    error_types = err.get("error_types", [])
    has_t1 = int("1" in error_types)
    has_t2 = int("2" in error_types)
    has_t3 = int("3" in error_types)
    num_errors = len([t for t in error_types if t in ["1","2","3"]])
    worst_opt = opt.get("worst_option_score", 2)

    c1_ok = int(r["pred_c1"] == r["correct_answer"])
    c2_ok = int(r["pred_c2"] == r["correct_answer"])

    rows.append({
        "id": qid,
        "Has_T1": has_t1,
        "Has_T2": has_t2,
        "Has_T3": has_t3,
        "Num_Error_Types": num_errors,
        "Option_Quality_Worst": worst_opt,
        "c1_ok": c1_ok,
        "c2_ok": c2_ok,
        "actual_failure": int(c1_ok == 1 and c2_ok == 0)
    })

df = pd.DataFrame(rows)

print(f"Total questions: {len(df)}")
print(f"Actual failures (C1✓ C2✗): {df['actual_failure'].sum()}/200")
print()

def evaluate_rule(pred, label):
    tp = ((pred == 1) & (df['actual_failure'] == 1)).sum()
    fp = ((pred == 1) & (df['actual_failure'] == 0)).sum()
    fn = ((pred == 0) & (df['actual_failure'] == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"{label:<35} P={prec:.2f} R={rec:.2f} F1={f1:.2f} (covers {pred.sum()})")

print("=== Filter Rule Validation (200 questions) ===")
evaluate_rule((df['Has_T1'] == 1).astype(int), "Only T1")
evaluate_rule((df['Has_T1'] == 1) & (df['Has_T3'] == 1), "T1 + T3")
evaluate_rule((df['Num_Error_Types'] >= 2).astype(int), "Num errors >= 2")
evaluate_rule((df['Num_Error_Types'] >= 3).astype(int), "Num errors >= 3")
evaluate_rule((df['Option_Quality_Worst'] > 2).astype(int), "Bad options (worst>2)")
evaluate_rule(
    ((df['Has_T1'] == 1) & (df['Num_Error_Types'] >= 2) & (df['Option_Quality_Worst'] > 2)).astype(int),
    "T1 + Num>=2 + Bad options"
)
evaluate_rule(
    ((df['Has_T1'] == 1) & (df['Option_Quality_Worst'] > 2)).astype(int),
    "T1 + Bad options"
)