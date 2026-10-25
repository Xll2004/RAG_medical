import json
import pandas as pd

# Load all data
with open("gold_set_200_translated.json") as f:
    questions = json.load(f)
with open("accuracy_results_200.json") as f:
    acc_data = json.load(f)
with open("error_annotations_200.json") as f:
    errors = json.load(f)
with open("question_classifications_200.json") as f:
    classifications = json.load(f)
with open("option_quality_analysis_200.json") as f:
    option_quality = json.load(f)
with open("selfcheck_scores_bertscore_200.json") as f:
    bert_scores = json.load(f)

# Build lookups
q_by_id = {q["id"]: q for q in questions}
err_by_id = {e["id"]: e for e in errors}
cls_by_id = {c["id"]: c for c in classifications}
opt_by_id = {o["id"]: o for o in option_quality}
bert_by_id = {s["id"]: s for s in bert_scores}

# Build dataframe
rows = []
for r in acc_data:
    qid = r["id"]
    q = q_by_id.get(qid, {})
    err = err_by_id.get(qid, {})
    cls = cls_by_id.get(qid, {})
    opt = opt_by_id.get(qid, {})
    bert = bert_by_id.get(qid, {})

    c1_ok = int(r["pred_c1"] == r["correct_answer"])
    c2_ok = int(r["pred_c2"] == r["correct_answer"])
    c3_ok = int(r["pred_c3"] == r["correct_answer"])
    c4_ok = int(r["pred_c4"] == r["correct_answer"])
    c5_ok = int(r["pred_c5"] == r["correct_answer"])

    error_types = err.get("error_types", [])

    rows.append({
        "ID": qid,
        "Question_ZH": q.get("question", ""),
        "Question_MT": q.get("question_mt", ""),
        "Correct_Answer": r["correct_answer"],
        "Category": cls.get("category", ""),
        "Has_T1": int("1" in error_types),
        "Has_T2": int("2" in error_types),
        "Has_T3": int("3" in error_types),
        "Num_Error_Types": len([t for t in error_types if t in ["1","2","3"]]),
        "Error_Explanation": err.get("explanation", ""),
        "Option_Quality_Avg": round(opt.get("avg_option_score", 0), 2),
        "Option_Quality_Worst": opt.get("worst_option_score", 0),
        "C1_Correct": c1_ok,
        "C2_Correct": c2_ok,
        "C3_Correct": c3_ok,
        "C4_Correct": c4_ok,
        "C5_Correct": c5_ok,
        "C1_C2_Gap": c1_ok - c2_ok,
        "C2_C5_Improvement": c5_ok - c2_ok,
        "BERTScore_C1": round(bert.get("score_c1", 0), 3),
        "BERTScore_C2": round(bert.get("score_c2", 0), 3),
        "BERTScore_C5": round(bert.get("score_c5", 0), 3),
        "Pattern": f"C1={'✓' if c1_ok else '✗'} C2={'✓' if c2_ok else '✗'} C3={'✓' if c3_ok else '✗'} C5={'✓' if c5_ok else '✗'}"
    })

df = pd.DataFrame(rows)

# Sort by C1-C2 gap (most impacted first)
df = df.sort_values("C1_C2_Gap", ascending=False)

# Save to Excel
with pd.ExcelWriter("experiment_results_200.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="All Questions", index=False)

    # Summary sheet
    summary_data = {
        "Metric": [
            "Total Questions",
            "C1 Accuracy", "C2 Accuracy", "C3 Accuracy", "C4 Accuracy", "C5 Accuracy",
            "C1-C2 Gap",
            "Questions with T1", "Questions with T2", "Questions with T3",
            "C1✓ C2✗ (Translation hurt)",
            "C1✓ C2✓ (Translation robust)",
            "All failed (C1✗ C2✗)",
        ],
        "Value": [
            len(df),
            f"{df['C1_Correct'].mean():.1%}",
            f"{df['C2_Correct'].mean():.1%}",
            f"{df['C3_Correct'].mean():.1%}",
            f"{df['C4_Correct'].mean():.1%}",
            f"{df['C5_Correct'].mean():.1%}",
            f"{(df['C1_Correct'].mean() - df['C2_Correct'].mean()):.1%}",
            f"{df['Has_T1'].sum()}/200",
            f"{df['Has_T2'].sum()}/200",
            f"{df['Has_T3'].sum()}/200",
            f"{((df['C1_Correct']==1) & (df['C2_Correct']==0)).sum()}/200",
            f"{((df['C1_Correct']==1) & (df['C2_Correct']==1)).sum()}/200",
            f"{((df['C1_Correct']==0) & (df['C2_Correct']==0)).sum()}/200",
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

    # Category breakdown
    cat_summary = df.groupby("Category").agg(
        Count=("ID", "count"),
        C1_Acc=("C1_Correct", "mean"),
        C2_Acc=("C2_Correct", "mean"),
        C5_Acc=("C5_Correct", "mean"),
        Gap=("C1_C2_Gap", "mean")
    ).round(3).reset_index()
    cat_summary.to_excel(writer, sheet_name="By Category", index=False)

print("Done. Saved to experiment_results_200.xlsx")
print(f"Total rows: {len(df)}")