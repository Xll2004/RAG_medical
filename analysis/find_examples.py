# find_examples.py
import json
gold = {q["id"]: q for q in json.load(open("data/gold_set_200_translated.json"))}
c1 = {r["id"]: r for r in json.load(open("results/results_condition1_200_clean.json"))}
c2 = {r["id"]: r for r in json.load(open("results/results_condition2_200.json"))}
main = {r["id"]: r for r in json.load(open("results/accuracy_results_200_clean.json"))}
mc = {k: {r["id"]: r for r in v} for k, v in json.load(open("results/results_matched_context_200.json")).items()}

ok = lambda cell, i: mc[cell][i]["pred"] == gold[i]["answer"]
ids = [i for i in main
       if ok("c2q_c1docs", i) and ok("c2q_random", i) and main[i]["pred_c2"] != gold[i]["answer"]]
print(f"{len(ids)} flip items (C2 query: correct with C1 docs and random, wrong with C2 docs)\n")

for i in ids[:8]:
    g, wrong = gold[i], main[i]["pred_c2"]
    print("=" * 90)
    print(f"id {i} | gold {g['answer']}: {g['option'][g['answer']]} | C2 chose {wrong}: {g['option'].get(wrong)}")
    print("ZH:", g["question"])
    print("C1:", c1[i]["question_en_baseline"])
    print("C2:", c2[i]["question_mt"])
    for k, p in enumerate(c2[i]["retrieved_docs"], 1):
        print(f"  C2 doc{k}: {p[:220]}")