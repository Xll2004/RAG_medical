# check_c2a_c3_vs_c2.py
import json
from math import comb

def norm(x):
    # Normalise answer format: "A", "A.", " a" -> "A"
    return str(x).strip().upper()[:1] if x is not None else ""

main = json.load(open("results/accuracy_results_200.json"))
c2a  = json.load(open("results/results_c2a_c3_200.json"))

c2_ok  = {r["id"]: norm(r["pred_c2"]) == norm(r["correct_answer"]) for r in main}
c2a_ok = {r["id"]: norm(r["answer"])  == norm(r["correct_answer"]) for r in c2a}

assert set(c2_ok) == set(c2a_ok), "id mismatch between files"
# The gold answers in both files must match
gold_main = {r["id"]: norm(r["correct_answer"]) for r in main}
gold_c2a  = {r["id"]: norm(r["correct_answer"]) for r in c2a}
assert gold_main == gold_c2a, "correct_answer mismatch between files"

qids = sorted(c2_ok)
b = sum(c2_ok[q] and not c2a_ok[q] for q in qids)   # C2 correct, C2a_C3 wrong
c = sum(not c2_ok[q] and c2a_ok[q] for q in qids)   # C2 wrong, C2a_C3 correct
n = b + c

p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)

print(f"n = {len(qids)}")
print(f"C2 acc     = {sum(c2_ok.values())/len(qids):.3f}")
print(f"C2a_C3 acc = {sum(c2a_ok.values())/len(qids):.3f}")
print(f"discordant: b = {b}, c = {c}")
print(f"exact McNemar p = {p:.6f}")

# Cross-check: c2a_mechanism_cases.json should be exactly these b questions
mech_ids = {r["id"] for r in json.load(open("results/c2a_mechanism_cases.json"))}
b_ids = {q for q in qids if c2_ok[q] and not c2a_ok[q]}
print(f"mechanism cases match b set: {mech_ids == b_ids} "
      f"(mech={len(mech_ids)}, b={len(b_ids)})")