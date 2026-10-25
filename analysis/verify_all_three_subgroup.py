import json
from statsmodels.stats.contingency_tables import mcnemar

ACCURACY_FILE   = "results/accuracy_results_200.json"
ANNOTATION_FILE = "results/error_annotations_200.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_correct(entry, condition):
    key  = f"pred_{condition.lower()}"
    pred = entry.get(key)
    gt   = entry.get("correct_answer")
    if pred is None or gt is None:
        return None
    return pred.strip().upper() == gt.strip().upper()

def get_error_types(annot_entry):
    types = annot_entry.get("error_types", [])
    return {
        "T1": "1" in types,
        "T2": "2" in types,
        "T3": "3" in types,
    }

def main():
    accuracy_data = load_json(ACCURACY_FILE)
    annot_data    = load_json(ANNOTATION_FILE)
    annot_by_id   = {str(entry["id"]): entry for entry in annot_data}

    all_three = []
    for entry in accuracy_data:
        qid = str(entry["id"])
        if qid not in annot_by_id:
            continue
        et = get_error_types(annot_by_id[qid])
        if et["T1"] and et["T2"] and et["T3"]:
            all_three.append(entry)

    n = len(all_three)
    print(f"\nAll-three subgroup size: n = {n}")

    c3_correct, c5_correct = [], []
    for entry in all_three:
        c3 = get_correct(entry, "C3")
        c5 = get_correct(entry, "C5")
        if c3 is None or c5 is None:
            print(f"  WARNING: id={entry['id']} has no prediction, skipped")
            continue
        c3_correct.append(c3)
        c5_correct.append(c5)

    valid_n = len(c3_correct)
    print(f"Valid n: {valid_n}")
    print(f"C3 accuracy: {sum(c3_correct)}/{valid_n} = {sum(c3_correct)/valid_n:.1%}")
    print(f"C5 accuracy: {sum(c5_correct)}/{valid_n} = {sum(c5_correct)/valid_n:.1%}")

    a = sum(1 for x,y in zip(c3_correct,c5_correct) if     x and     y)
    b = sum(1 for x,y in zip(c3_correct,c5_correct) if     x and not y)
    c = sum(1 for x,y in zip(c3_correct,c5_correct) if not x and     y)
    d = sum(1 for x,y in zip(c3_correct,c5_correct) if not x and not y)

    print(f"\nContingency table:\n         C5✓  C5✗\n  C3✓   {a:3d}  {b:3d}\n  C3✗   {c:3d}  {d:3d}")

    if b + c == 0:
        print("\nMcNemar: p = 1.000 (no discordant pairs)")
    else:
        result = mcnemar([[a,b],[c,d]], exact=True)
        print(f"\nMcNemar's test (exact): p = {result.pvalue:.4f}")
        print("-> Difference not significant; sampling variance explanation holds" if result.pvalue >= 0.05 else "-> Difference significant; the cause needs careful discussion")

    print("\n── Per-question details ──")
    print(f"{'ID':>6}  C3  C5")
    for entry, c3, c5 in zip(all_three[:valid_n], c3_correct, c5_correct):
        print(f"{entry['id']:>6}  {'✓' if c3 else '✗'}   {'✓' if c5 else '✗'}")

if __name__ == "__main__":
    main()
