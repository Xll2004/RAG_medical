import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

with open("results/accuracy_results_200.json") as f:
    main_results = json.load(f)

with open("results/results_norag_baseline_200.json") as f:
    norag_results = json.load(f)

# Align by ID to be safe
main_by_id = {r["id"]: r for r in main_results}
norag_by_id = {r["id"]: r for r in norag_results}

common_ids = [i for i in main_by_id if i in norag_by_id]
print(f"Aligned questions: {len(common_ids)}")

c1 = [int(main_by_id[i]["pred_c1"] == main_by_id[i]["correct_answer"]) for i in common_ids]
c2 = [int(main_by_id[i]["pred_c2"] == main_by_id[i]["correct_answer"]) for i in common_ids]
norag = [int(norag_by_id[i]["pred_norag"] == norag_by_id[i]["correct_answer"]) for i in common_ids]

print("\n=== Accuracy ===")
print(f"No-RAG:  {np.mean(norag):.3f}")
print(f"C1:      {np.mean(c1):.3f}")
print(f"C2:      {np.mean(c2):.3f}")


def run_mcnemar(a, b, label):
    both_correct = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    both_wrong = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=False)
    sig = "✅ significant" if result.pvalue < 0.05 else "❌ not significant"
    print(f"{label}: p = {result.pvalue:.4f} {sig}  (discordant: a_only={a_only}, b_only={b_only})")


print("\n=== McNemar Test ===")
run_mcnemar(norag, c1, "No-RAG vs C1 (retrieval's marginal contribution given good translation)")
run_mcnemar(norag, c2, "No-RAG vs C2 (corrupted retrieval vs no retrieval at all)")