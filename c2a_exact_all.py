# c2a_exact_all.py
import json, glob
from math import comb

def norm(x):
    return str(x).strip().upper()[:1] if x is not None else ""

def exact_p(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n) if n else 1.0

main = {r["id"]: r for r in json.load(open("accuracy_results_200_clean.json"))}
gold = {i: norm(r["correct_answer"]) for i, r in main.items()}
ok = {c: {i: norm(r[f"pred_{c}"]) == gold[i] for i, r in main.items()}
      for c in ["c1", "c2", "c3", "c5"]}

c2a_names = []
for fn in sorted(glob.glob("results_c2a_*_200.json")):
    name = fn[len("results_"):-len("_200.json")]
    d = json.load(open(fn))
    assert {r["id"] for r in d} == set(main), f"{fn}: id mismatch"
    assert all(norm(r["correct_answer"]) == gold[r["id"]] for r in d), f"{fn}: gold mismatch"
    ok[name] = {r["id"]: norm(r["answer"]) == gold[r["id"]] for r in d}
    c2a_names.append(name)

ids = sorted(main)
print("acc: " + "  ".join(f"{k}={sum(v.values())/len(ids):.3f}" for k, v in ok.items()))
print(f"\n{'pair':22s} {'b':>4s} {'c':>4s} {'exact p':>10s}")
for a in c2a_names:
    for ref in ["c1", "c2", "c3", "c5"]:
        b = sum(ok[ref][q] and not ok[a][q] for q in ids)   # ref 对, C2a 错
        c = sum(ok[a][q] and not ok[ref][q] for q in ids)   # C2a 对, ref 错
        print(f"{ref.upper()+' vs '+a:22s} {b:>4d} {c:>4d} {exact_p(b, c):>10.6f}")