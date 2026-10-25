"""
Prepare the human annotation set -- independent human validation of the error taxonomy
"""

import json
import random
import csv

# ── Config ──────────────────────────────────────────────────
GOLD_FILE       = "data/gold_set_200_translated.json"
ANNOTATION_FILE = "results/error_annotations_200.json"
SAMPLE_SIZE     = 60
RANDOM_SEED     = 2026
OUTPUT_BLIND    = "data/human_annotation_blind.csv"
OUTPUT_KEY      = "data/human_annotation_answer_key.json"
# ──────────────────────────────────────────────────────────


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    gold_data  = load_json(GOLD_FILE)
    annot_data = load_json(ANNOTATION_FILE)

    annot_by_id = {str(e["id"]): e for e in annot_data}

    valid_ids = [str(e["id"]) for e in gold_data if str(e["id"]) in annot_by_id]
    print(f"Total eligible questions: {len(valid_ids)}")

    random.seed(RANDOM_SEED)
    sampled_ids = sorted(random.sample(valid_ids, SAMPLE_SIZE))
    print(f"Sampled {len(sampled_ids)} questions for human annotation.")

    gold_by_id = {str(e["id"]): e for e in gold_data}

    shuffled_ids = sampled_ids.copy()
    random.shuffle(shuffled_ids)

    with open(OUTPUT_BLIND, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_no",
            "question_zh",
            "option_A_zh", "option_B_zh", "option_C_zh", "option_D_zh", "option_E_zh",
            "question_mt",
            "option_A_mt", "option_B_mt", "option_C_mt", "option_D_mt", "option_E_mt",
            "T1_terminology",
            "T2_syntactic",
            "T3_semantic",
            "annotator_notes",
        ])

        for item_no, qid in enumerate(shuffled_ids, start=1):
            g = gold_by_id[qid]
            opt_zh = g.get("option", {})
            opt_mt = g.get("option_mt", {})
            writer.writerow([
                item_no,
                g.get("question", ""),
                opt_zh.get("A", ""), opt_zh.get("B", ""), opt_zh.get("C", ""),
                opt_zh.get("D", ""), opt_zh.get("E", ""),
                g.get("question_mt", ""),
                opt_mt.get("A", ""), opt_mt.get("B", ""), opt_mt.get("C", ""),
                opt_mt.get("D", ""), opt_mt.get("E", ""),
                "", "", "", "",
            ])

    print(f"\n✓ Blind annotation file written: {OUTPUT_BLIND}")
    print("  -> Send this file to each annotator (one independent copy each)")
    print("  -> Annotators see only item_no, not the original question id or GPT-4o-mini's labels")

    answer_key = []
    for item_no, qid in zip(range(1, len(shuffled_ids) + 1), shuffled_ids):
        a = annot_by_id[qid]
        types = a.get("error_types", [])
        answer_key.append({
            "item_no": item_no,
            "original_id": int(qid),
            "gpt4o_mini_T1": "1" in types,
            "gpt4o_mini_T2": "2" in types,
            "gpt4o_mini_T3": "3" in types,
        })

    with open(OUTPUT_KEY, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, indent=2, ensure_ascii=False)

    print(f"✓ Answer key written: {OUTPUT_KEY}")
    print("  -> Keep this file; do not send it to annotators. Used later for human-vs-GPT agreement")

    print(f"\nNext steps:")
    print(f"  1. Copy {OUTPUT_BLIND} to annotator1.csv, annotator2.csv (more copies for more annotators)")
    print(f"  2. Annotators independently fill in the T1/T2/T3 columns (1 = error present, 0 = absent)")
    print(f"  3. After annotation, compute agreement with compute_agreement.py:")
    print(f"     - Human vs Human (Cohen's kappa / Jaccard)")
    print(f"     - Human vs GPT-4o-mini (Jaccard, against {OUTPUT_KEY})")


if __name__ == "__main__":
    main()
