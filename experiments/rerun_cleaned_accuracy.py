import ast, json, os, time
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_func(path, name):
    tree = ast.parse(open(path, encoding="utf-8").read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"client": client}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns[name]


gold = {q["id"]: q for q in json.load(open("data/gold_set_200_translated.json", encoding="utf-8"))}
N_RUNS = 3

JOBS = [
    # (clean results file, question field, generator source, original accuracy file, output file)
    ("results/results_condition1_200_clean.json", "question_en_baseline", "experiments/run_accuracy.py",
     "results/accuracy_results_200.json", "results/accuracy_results_200_clean.json"),
    ("results/results_c1prime_200_clean.json", "question_en_gpt41", "experiments/score_c1prime.py",
     "results/accuracy_c1prime_200.json", "results/accuracy_c1prime_200_clean.json"),
]

for clean_fn, qkey, script, acc_fn, out_fn in JOBS:
    get_answer_choice = load_func(script, "get_answer_choice")
    items = [x for x in json.load(open(clean_fn, encoding="utf-8")) if x.get("cleaned")]

    acc = json.load(open(acc_fn, encoding="utf-8"))
    pred_key = "pred_c1" if acc_fn == "results/accuracy_results_200.json" else \
        next(k for k in acc[0] if k.startswith("pred"))
    by_id = {r["id"]: r for r in acc}

    # 3 runs, same order as original scripts: outer loop runs, inner loop questions
    votes = {x["id"]: [] for x in items}
    for run in range(N_RUNS):
        for x in items:
            pred = get_answer_choice(x[qkey], gold[x["id"]]["option"], x["retrieved_docs"])
            votes[x["id"]].append(pred)
            time.sleep(0.2)
        print(f"{clean_fn} run {run+1} done")

    flips = []
    for qid, v in votes.items():
        r = by_id[qid]
        new = Counter(v).most_common(1)[0][0]
        ans = gold[qid]["answer"]
        r[pred_key + "_raw"] = r[pred_key]
        r[pred_key] = new
        r["cleaned_votes"] = v
        if (r[pred_key + "_raw"] == ans) != (new == ans):
            flips.append((qid, r[pred_key + "_raw"], new, ans))

    json.dump(acc, open(out_fn, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n = len(acc)
    old_acc = sum(r.get(pred_key + "_raw", r[pred_key]) == gold[r["id"]]["answer"] for r in acc) / n
    new_acc = sum(r[pred_key] == gold[r["id"]]["answer"] for r in acc) / n
    print(f"\n=== {out_fn} [{pred_key}] ===")
    print(f"re-run {len(items)} questions | accuracy {old_acc:.3f} -> {new_acc:.3f}")
    print(f"correctness flips ({len(flips)}):")
    for qid, o, nw, ans in flips:
        print(f"  id={qid}: {o} -> {nw} (gold {ans})  {'gain' if nw == ans else 'loss'}")
