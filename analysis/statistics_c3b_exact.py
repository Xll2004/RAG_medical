import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# Load accuracy results
with open("results/accuracy_results_200.json") as f:
    acc_data = json.load(f)

with open("results/accuracy_results_c3b_200.json") as f:
    c3b_data = json.load(f)

# Build arrays
c2 = [int(r["pred_c2"] == r["correct_answer"]) for r in acc_data]
c3 = [int(r["pred_c3"] == r["correct_answer"]) for r in acc_data]
c3b = [int(r["pred_c3b"] == r["correct_answer"]) for r in c3b_data]

print(f"C2 MT:           {np.mean(c2):.3f}")
print(f"C3 General:      {np.mean(c3):.3f}")
print(f"C3b Terminology: {np.mean(c3b):.3f}")
print()

def run_mcnemar(a, b, label):
    both_correct = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    both_wrong = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=True)
    sig = "✅ significant" if result.pvalue < 0.05 else "❌ not significant"
    print(f"{label}: p = {result.pvalue:.4f} {sig}")

print("=== McNemar Test ===")
run_mcnemar(c2, c3b, "C2 vs C3b (MT vs terminology-only)")
run_mcnemar(c3, c3b, "C3 vs C3b (general vs terminology-only)")
run_mcnemar(c2, c3,  "C2 vs C3  (MT vs general correction)")