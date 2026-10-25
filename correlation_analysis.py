import json
import numpy as np
from scipy import stats

# Load accuracy results
with open("accuracy_results_200.json") as f:
    acc_data = json.load(f)

# Load BERTScore results
with open("selfcheck_scores_bertscore_200.json") as f:
    bert_data = json.load(f)

# Per-condition accuracy
c1_acc = [int(r["pred_c1"] == r["correct_answer"]) for r in acc_data]
c2_acc = [int(r["pred_c2"] == r["correct_answer"]) for r in acc_data]
c3_acc = [int(r["pred_c3"] == r["correct_answer"]) for r in acc_data]
c4_acc = [int(r["pred_c4"] == r["correct_answer"]) for r in acc_data]
c5_acc = [int(r["pred_c5"] == r["correct_answer"]) for r in acc_data]

# Per-condition BERTScore (lower = less hallucination = better)
c1_bert = [s["score_c1"] for s in bert_data]
c2_bert = [s["score_c2"] for s in bert_data]
c3_bert = [s["score_c3"] for s in bert_data]
c4_bert = [s["score_c4"] for s in bert_data]
c5_bert = [s["score_c5"] for s in bert_data]

# Condition-level correlation (5 data points)
acc_means = [np.mean(c1_acc), np.mean(c2_acc), np.mean(c3_acc), np.mean(c4_acc), np.mean(c5_acc)]
bert_means = [np.mean(c1_bert), np.mean(c2_bert), np.mean(c3_bert), np.mean(c4_bert), np.mean(c5_bert)]

r, p = stats.pearsonr(acc_means, bert_means)
print("=== Condition-level Correlation (200 questions) ===")
print(f"Accuracy means:    {[f'{x:.3f}' for x in acc_means]}")
print(f"BERTScore means:   {[f'{x:.3f}' for x in bert_means]}")
print(f"Pearson r = {r:.3f}, p = {p:.4f}")
print()

# Question-level correlation (200 data points, averaged across conditions)
acc_per_q = [(a+b+c+d+e)/5 for a,b,c,d,e in zip(c1_acc, c2_acc, c3_acc, c4_acc, c5_acc)]
bert_per_q = [(a+b+c+d+e)/5 for a,b,c,d,e in zip(c1_bert, c2_bert, c3_bert, c4_bert, c5_bert)]

r2, p2 = stats.pearsonr(acc_per_q, bert_per_q)
print("=== Question-level Correlation (200 questions) ===")
print(f"Pearson r = {r2:.3f}, p = {p2:.4f}")