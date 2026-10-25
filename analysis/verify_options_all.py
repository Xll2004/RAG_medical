# verify_options_all.py -- for each condition, determine whether the archived predictions were scored with the Chinese options or the translated/corrected options
# For each condition: same question + same retrieved docs, paired with (a) the Chinese options and (b) the English options generated for that condition, one run each,
# compared question-by-question with the archived predictions. Whichever option set agrees at ~97% (the run-to-run noise level) is the one the archive was produced with.
import ast, json, os, time
from concurrent.futures import ThreadPoolExecutor
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

get_answer_choice = load_func("experiments/run_accuracy.py", "get_answer_choice")
gold = json.load(open("data/gold_set_200_translated.json", encoding="utf-8"))
ids = [q["id"] for q in gold]

def load(fn): return json.load(open(fn, encoding="utf-8"))

c1 = load("results/results_condition1_200.json")   # Original (uncleaned) C1, corresponding to pred_c1 from Jun 3
c2 = load("results/results_condition2_200.json")

# (name, result list, question field, English-option source lambda i, archive file, archive field)
CONDS = [
    ("C1",  c1, "question_en_baseline", lambda i: c1[i]["options_en"],       "results/accuracy_results_200.json",     "pred_c1"),
    ("C2",  c2, "question_mt",          lambda i: gold[i]["option_mt"],      "results/accuracy_results_200.json",     "pred_c2"),
]
for name, fn, pk in [("C3", "results/results_condition3_200.json", "pred_c3"),
                     ("C3b", "results/results_condition3b_200.json", "pred_c3b"),
                     ("C4", "results/results_condition4_200.json", "pred_c4"),
                     ("C5", "results/results_condition5_200.json", "pred_c5")]:
    r = load(fn)
    acc_fn = "results/accuracy_results_c3b_200.json" if name == "C3b" else "results/accuracy_results_200.json"
    CONDS.append((name, r, "question_corrected", (lambda r: lambda i: r[i]["options_corrected"])(r), acc_fn, pk))
c1p = load("results/results_c1prime_200.json")
CONDS.append(("C1prime", c1p, "question_en_gpt41", lambda i: c1p[i]["options_en"], "results/accuracy_c1prime_200.json", "pred_c1prime"))

def call(q, opts, docs):
    for t in range(5):
        try:
            return get_answer_choice(q, opts, docs)
        except Exception:
            time.sleep(2 * (t + 1))
    return "ERR"

OUT = "results/verify_options_all_out.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
print(f"{'cond':8s} {'options':10s} {'acc':>6s} {'stored acc':>10s} {'agree':>8s}")
for name, res, qkey, en_opts, acc_fn, pk in CONDS:
    assert [x["id"] for x in res[:200]] == ids, f"{name} ids not aligned"
    stored = {r["id"]: r[pk] for r in load(acc_fn)}
    st_acc = sum(stored[q["id"]] == q["answer"] for q in gold) / 200
    for label, optf in [("chinese", lambda i: gold[i]["option"]), ("english", en_opts)]:
        key = f"{name}_{label}"
        if key not in out:
            with ThreadPoolExecutor(8) as ex:
                out[key] = list(ex.map(lambda i: call(res[i][qkey], optf(i), res[i]["retrieved_docs"]), range(200)))
            json.dump(out, open(OUT, "w"), indent=1)
        preds = out[key]
        acc = sum(p == gold[i]["answer"] for i, p in enumerate(preds)) / 200
        agree = sum(p == stored[ids[i]] for i, p in enumerate(preds))
        errs = preds.count("ERR")
        print(f"{name:8s} {label:10s} {acc:6.3f} {st_acc:10.3f} {agree:5d}/200" + (f"  ({errs} ERR)" if errs else ""))
