import json

with open("gold_set_200_translated.json") as f:
    questions = json.load(f)
with open("results_condition1_200.json") as f:
    c1 = json.load(f)
with open("results_condition2_200.json") as f:
    c2 = json.load(f)
with open("results_condition3_200.json") as f:
    c3 = json.load(f)
with open("results_condition5_200.json") as f:
    c5 = json.load(f)
with open("accuracy_results_200.json") as f:
    acc_data = json.load(f)
with open("error_annotations_200.json") as f:
    errors = json.load(f)
with open("question_classifications_200.json") as f:
    classifications = json.load(f)

q_by_id = {q["id"]: q for q in questions}
c1_by_id = {r["id"]: r for r in c1}
c2_by_id = {r["id"]: r for r in c2}
c3_by_id = {r["id"]: r for r in c3}
c5_by_id = {r["id"]: r for r in c5}
err_by_id = {e["id"]: e for e in errors}
cls_by_id = {c["id"]: c for c in classifications}

# Target IDs from qualitative analysis
target_ids = [2417, 1758, 5868, 1939, 4353]

for qid in target_ids:
    q = q_by_id[qid]
    err = err_by_id.get(qid, {})
    cls = cls_by_id.get(qid, {})
    acc = next((r for r in acc_data if r["id"] == qid), {})

    c1_ok = acc.get("pred_c1") == q["answer"]
    c2_ok = acc.get("pred_c2") == q["answer"]
    c3_ok = acc.get("pred_c3") == q["answer"]
    c5_ok = acc.get("pred_c5") == q["answer"]

    print(f"{'='*70}")
    print(f"ID: {qid} | Category: {cls.get('category')} | Answer: {q['answer']}")
    print(f"Pattern: C1={'Correct' if c1_ok else 'Wrong'} C2={'Correct' if c2_ok else 'Wrong'} C3={'Correct' if c3_ok else 'Wrong'} C5={'Correct' if c5_ok else 'Wrong'}")
    print(f"Error types: {err.get('error_types')} | {err.get('explanation','')[:100]}")
    print()
    print(f"ZH:  {q['question']}")
    print(f"MT:  {q['question_mt']}")
    print(f"C3:  {c3_by_id[qid]['question_corrected']}")
    print(f"C5:  {c5_by_id[qid]['question_corrected']}")
    print()
    print("Options ZH:", q['option'])
    print("Options MT:", q['option_mt'])
    print("Options C3:", c3_by_id[qid]['options_corrected'])
    print("Options C5:", c5_by_id[qid]['options_corrected'])
    print()