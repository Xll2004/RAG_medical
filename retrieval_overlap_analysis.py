import json
from collections import defaultdict

with open("results_condition1_200.json") as f:
    c1 = json.load(f)
with open("results_condition2_200.json") as f:
    c2 = json.load(f)

jaccard_scores = []
overlap_counts = []
per_question = []

c1_map = {q["id"]: set(q["retrieved_docs"]) for q in c1}
c2_map = {q["id"]: set(q["retrieved_docs"]) for q in c2}
acc_map = {q["id"]: int(q["answer"] == q["correct_answer"]) for q in c2}

for qid in c1_map:
    if qid not in c2_map:
        continue
    s1 = c1_map[qid]
    s2 = c2_map[qid]
    intersection = s1 & s2
    union = s1 | s2
    jaccard = len(intersection) / len(union) if union else 0
    jaccard_scores.append(jaccard)
    overlap_counts.append(len(intersection))
    per_question.append({"id": qid, "jaccard": jaccard, "overlap": len(intersection)})

n = len(jaccard_scores)
avg_jaccard = sum(jaccard_scores) / n
avg_overlap = sum(overlap_counts) / n
zero_overlap = sum(1 for j in jaccard_scores if j == 0)
full_overlap = sum(1 for j in jaccard_scores if j == 1.0)

print(f"N questions: {n}")
print(f"Avg Jaccard (C1 vs C2): {avg_jaccard:.3f}")
print(f"Avg doc overlap (out of 3): {avg_overlap:.3f}")
print(f"Zero overlap: {zero_overlap} ({zero_overlap/n*100:.1f}%)")
print(f"Full overlap: {full_overlap} ({full_overlap/n*100:.1f}%)")
for k in [0, 1, 2, 3]:
    count = sum(1 for o in overlap_counts if o == k)
    print(f"  {k}/3 docs overlap: {count} ({count/n*100:.1f}%)")

print("\nC2 accuracy by retrieval overlap:")
groups = defaultdict(list)
for q in per_question:
    if q["id"] in acc_map:
        groups[q["overlap"]].append(acc_map[q["id"]])
for k in sorted(groups):
    accs = groups[k]
    print(f"  overlap={k}: n={len(accs)}, accuracy={sum(accs)/len(accs)*100:.1f}%")

with open("retrieval_overlap_200.json", "w") as f:
    json.dump({"per_question": per_question, "summary": {"avg_jaccard": avg_jaccard, "avg_overlap": avg_overlap}}, f, indent=2)
print("\nSaved retrieval_overlap_200.json")
