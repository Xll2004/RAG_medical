import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

with open("results/accuracy_results_200_clean.json") as f:
    results = json.load(f)

c1 = [int(r["pred_c1"] == r["correct_answer"]) for r in results]
c2 = [int(r["pred_c2"] == r["correct_answer"]) for r in results]
c3 = [int(r["pred_c3"] == r["correct_answer"]) for r in results]
c4 = [int(r["pred_c4"] == r["correct_answer"]) for r in results]
c5 = [int(r["pred_c5"] == r["correct_answer"]) for r in results]

print("=== Accuracy (200 questions) ===")
print(f"C1 Baseline:  {np.mean(c1):.3f}")
print(f"C2 MT:        {np.mean(c2):.3f}")
print(f"C3 GPT-4o:    {np.mean(c3):.3f}")
print(f"C4 DeepSeek:  {np.mean(c4):.3f}")
print(f"C5 GPT-5.5:   {np.mean(c5):.3f}")
print()

def run_mcnemar(a, b, label):
    both_correct = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    both_wrong = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=False)
    sig = "✅ significant" if result.pvalue < 0.05 else "❌ not significant"
    print(f"{label}: p = {result.pvalue:.4f} {sig}")

print("=== McNemar Test ===")
run_mcnemar(c1, c2, "C1 vs C2 (Baseline vs MT)")
run_mcnemar(c1, c3, "C1 vs C3 (Baseline vs GPT-4o)")
run_mcnemar(c1, c4, "C1 vs C4 (Baseline vs DeepSeek)")
run_mcnemar(c1, c5, "C1 vs C5 (Baseline vs GPT-5.5)")
run_mcnemar(c2, c3, "C2 vs C3 (MT vs GPT-4o)")
run_mcnemar(c2, c4, "C2 vs C4 (MT vs DeepSeek)")
run_mcnemar(c2, c5, "C2 vs C5 (MT vs GPT-5.5)")
run_mcnemar(c3, c4, "C3 vs C4 (GPT-4o vs DeepSeek)")
run_mcnemar(c3, c5, "C3 vs C5 (GPT-4o vs GPT-5.5)")
run_mcnemar(c4, c5, "C4 vs C5 (DeepSeek vs GPT-5.5)")