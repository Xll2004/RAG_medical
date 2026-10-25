import json

with open("error_annotations_200.json") as f:
    errors = json.load(f)

with open("gold_set_200_translated.json") as f:
    questions = json.load(f)

q_by_id = {q["id"]: q for q in questions}

# Find questions with T3 error
t3_cases = [e for e in errors if "3" in e["error_types"]]
print(f"Total T3 cases: {len(t3_cases)}")
print()

for e in t3_cases[:10]:
    q = q_by_id[e["id"]]
    print(f"ID {e['id']}:")
    print(f"  ZH:  {q['question'][:60]}")
    print(f"  MT:  {q['question_mt'][:60]}")
    print(f"  Error: {e['explanation'][:80]}")
    print()