import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

with open("accuracy_results_200.json") as f:
    main_results = json.load(f)
with open("results_norag_baseline_200.json") as f:
    norag_results = json.load(f)
with open("results_c2a_c3_200.json") as f:
    c2a_c3 = json.load(f)
with open("results_condition1_200.json") as f:
    c1_full = json.load(f)
with open("results_condition2_200.json") as f:
    c2_full = json.load(f)

main_by_id = {r["id"]: r for r in main_results}
norag_by_id = {r["id"]: r for r in norag_results}
c2a_by_id = {r["id"]: r for r in c2a_c3}
c1_by_id = {r["id"]: r for r in c1_full}
c2_by_id = {r["id"]: r for r in c2_full}

ids = [i for i in main_by_id if i in norag_by_id and i in c2a_by_id]
print(f"Aligned questions: {len(ids)}\n")

c1_v = [int(main_by_id[i]["pred_c1"] == main_by_id[i]["correct_answer"]) for i in ids]
c2_v = [int(main_by_id[i]["pred_c2"] == main_by_id[i]["correct_answer"]) for i in ids]
norag_v = [int(norag_by_id[i]["pred_norag"] == norag_by_id[i]["correct_answer"]) for i in ids]
c2a_v = [int(c2a_by_id[i]["correct"]) for i in ids]

print("=== Accuracy ladder ===")
print(f"C1     (good query + matched docs):     {np.mean(c1_v):.3f}")
print(f"No-RAG (good query + no docs):          {np.mean(norag_v):.3f}")
print(f"C2     (bad query  + docs matched to it): {np.mean(c2_v):.3f}")
print(f"C2a_C3 (good query + docs from bad query): {np.mean(c2a_v):.3f}")


def run_mcnemar(a, b, label):
    both = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    neither = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
    res = mcnemar([[both, a_only], [b_only, neither]], exact=True)
    sig = "significant" if res.pvalue < 0.05 else "not significant"
    print(f"{label}\n    p = {res.pvalue:.4f} ({sig}); discordant: a_only={a_only}, b_only={b_only}")


print("\n=== McNemar tests ===")
run_mcnemar(norag_v, c2a_v, "No-RAG vs C2a_C3 (same query quality; docs vs no docs)")
run_mcnemar(c2_v, c2a_v, "C2 vs C2a_C3 (same docs; bad query vs corrected query)")

# Key diagnostic subset: questions C2 got right but C2a_C3 got wrong.
# Correcting the query made things WORSE while holding documents fixed.
flipped = [i for i in ids if main_by_id[i]["pred_c2"] == main_by_id[i]["correct_answer"]
           and c2a_by_id[i]["correct"] == 0]
print(f"\n=== Diagnostic subset: C2 correct -> C2a_C3 wrong (n={len(flipped)}) ===")

# Of these, how many did the no-RAG baseline get right?
# If no-RAG gets them right, the documents (not the query) are doing the damage.
norag_right_in_flipped = sum(1 for i in flipped if norag_by_id[i]["pred_norag"] == norag_by_id[i]["correct_answer"])
print(f"Of these {len(flipped)}, no-RAG answers {norag_right_in_flipped} correctly "
      f"({norag_right_in_flipped / len(flipped):.1%}) with no context at all.")
print("-> High rate here = the mismatched documents are actively misleading the generator,")
print("   since the same generator with the same-quality query succeeds when given no documents.\n")

# Dump cases for qualitative inspection
cases = []
for i in flipped:
    cases.append({
        "id": i,
        "question_c1_translation": c1_by_id[i].get("question_en_baseline", ""),
        "question_c2_mt": c2_by_id[i].get("question_mt", ""),
        "question_corrected_c2a": c2a_by_id[i]["question_corrected"],
        "retrieved_docs_from_c2_query": [d[:400] for d in c2a_by_id[i]["retrieved_docs"]],
        "correct_answer": c2a_by_id[i]["correct_answer"],
        "pred_c2": main_by_id[i]["pred_c2"],
        "pred_c2a_c3": c2a_by_id[i]["answer"],
        "pred_norag": norag_by_id[i]["pred_norag"],
        "votes_c2a_c3": c2a_by_id[i]["votes"],
    })

with open("c2a_mechanism_cases.json", "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(cases)} cases to c2a_mechanism_cases.json for qualitative review.")
print("\n=== First 3 cases (preview) ===")
for c in cases[:3]:
    print(f"\n--- id {c['id']} | gold={c['correct_answer']} | "
          f"C2={c['pred_c2']} (correct) | C2a_C3={c['pred_c2a_c3']} | no-RAG={c['pred_norag']}")
    print(f"  MT query (C2):        {c['question_c2_mt'][:120]}")
    print(f"  Corrected query (C2a): {c['question_corrected_c2a'][:120]}")
    print(f"  Doc 1 (retrieved by the MT query): {c['retrieved_docs_from_c2_query'][0][:200]}")
