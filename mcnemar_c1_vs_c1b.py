"""
McNemar's test comparing C1a (original C1, GPT-4o-mini generation)
against C1b (control, Claude generation on the same translations).
"""

import json
from statsmodels.stats.contingency_tables import mcnemar

with open("accuracy_results_200.json", "r", encoding="utf-8") as f:
    accuracy_results = json.load(f)

with open("results_c1b_control.json", "r", encoding="utf-8") as f:
    c1b = json.load(f)

# Build lookup by id
c1a_by_id = {item["id"]: item for item in accuracy_results}
c1b_by_id = {item["id"]: item for item in c1b}

common_ids = set(c1a_by_id.keys()) & set(c1b_by_id.keys())
print(f"Comparing {len(common_ids)} questions present in both conditions.")

# Contingency table cells
both_correct = 0
a_correct_b_wrong = 0
a_wrong_b_correct = 0
both_wrong = 0

for qid in common_ids:
    item_a = c1a_by_id[qid]
    item_b = c1b_by_id[qid]

    a_correct = item_a["pred_c1"] == item_a["correct_answer"]
    b_correct = item_b["correct"]

    if a_correct and b_correct:
        both_correct += 1
    elif a_correct and not b_correct:
        a_correct_b_wrong += 1
    elif not a_correct and b_correct:
        a_wrong_b_correct += 1
    else:
        both_wrong += 1

table = [[both_correct, a_correct_b_wrong],
         [a_wrong_b_correct, both_wrong]]

print("\nContingency table:")
print(f"                  C1b correct   C1b wrong")
print(f"C1a correct       {both_correct:<13} {a_correct_b_wrong}")
print(f"C1a wrong         {a_wrong_b_correct:<13} {both_wrong}")

result = mcnemar(table, exact=True)
print(f"\nMcNemar's test: statistic={result.statistic}, p-value={result.pvalue:.4f}")

if result.pvalue < 0.05:
    print("=> Statistically significant difference between C1a and C1b.")
else:
    print("=> No statistically significant difference between C1a and C1b.")