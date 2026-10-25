import json

with open("results/spotcheck_agreement.json", "r", encoding="utf-8") as f:
    data = json.load(f)

jaccard_scores = []
for r in data["results"]:
    gpt = set(r["gpt_annotation"]) - {"s"}
    ds = set(r["deepseek_annotation"]) - {"s"}
    if not gpt and not ds:
        jaccard_scores.append(1.0)
    elif not gpt or not ds:
        jaccard_scores.append(0.0)
    else:
        intersection = len(gpt & ds)
        union = len(gpt | ds)
        jaccard_scores.append(intersection / union)

avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
print(f"Average Jaccard similarity: {avg_jaccard:.3f}")
