import json
import random

# Load all questions
with open("CMB/data/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

with open("CMB/data/CMB-test-choice-answer.json", "r", encoding="utf-8") as f:
    answers = json.load(f)

# Merge answers
answer_dict = {item["id"]: item["answer"] for item in answers}
for q in questions:
    q["answer"] = answer_dict[q["id"]]

# Filter single-choice only
single_choice = [q for q in questions if len(q["answer"]) == 1]
print(f"Total single-choice questions: {len(single_choice)}")

# Sample 200
random.seed(42)
sample_200 = random.sample(single_choice, 200)

# Save
with open("data/gold_set_200.json", "w", encoding="utf-8") as f:
    json.dump(sample_200, f, ensure_ascii=False, indent=2)

print(f"Sampled 200 single-choice questions")

# Check subject distribution
from collections import Counter
subjects = Counter(q["exam_subject"] for q in sample_200)
print(f"Total subjects: {len(subjects)}")
print("Top 10 subjects:")
for subject, count in sorted(subjects.items(), key=lambda x: -x[1])[:10]:
    print(f"  {subject}: {count}")