# review2_common.py -- shared code for the second-round review experiments (M2/M3/M4)
# Generation prompt taken verbatim from run_accuracy.py; 3 runs + majority vote; each cell is saved as soon as it finishes, resumable
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import ast, json, random, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60)
L = lambda f: json.load(open(f, encoding="utf-8"))


def load_func(path, name):
    """Extract only the function definitions from the original script, so the prompt is verbatim identical to the original experiment"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"client": client}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns[name]


_answer = load_func("experiments/run_accuracy.py", "get_answer_choice")


def answer(question, options, docs):
    for attempt in range(6):
        try:
            return _answer(question, options, docs)
        except Exception as e:
            wait = 2 ** attempt * 3
            print(f"  API error ({type(e).__name__}), retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("API failed after retries")


def chat(prompt, max_tokens=5, model="gpt-4o-mini"):
    for attempt in range(6):
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                               max_tokens=max_tokens, temperature=0)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            wait = 2 ** attempt * 3
            print(f"  API error ({type(e).__name__}), retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("API failed after retries")


QUESTIONS = L("data/gold_set_200_translated.json")
N = len(QUESTIONS)
RES = {c: L(f"results/results_condition{c}_200.json") for c in [2]}
RES[1] = L("results/results_condition1_200_clean.json")
for c, d in RES.items():
    assert [x["id"] for x in d[:N]] == [q["id"] for q in QUESTIONS], f"C{c} ids not aligned"
C1Q = [RES[1][i]["question_en_baseline"] for i in range(N)]
C2Q = [RES[2][i]["question_mt"] for i in range(N)]
ZHQ = [q["question"] for q in QUESTIONS]
C1DOCS = [RES[1][i]["retrieved_docs"] for i in range(N)]
C2DOCS = [RES[2][i]["retrieved_docs"] for i in range(N)]


def random_ctx(pool_lists, seed=42):
    """Same rule as matched_context_ablation.py: draw 3 passages from all retrieved passages, excluding those retrieved for this question"""
    rng = random.Random(seed)
    pool = sorted({d for lst in pool_lists for item in lst for d in item})
    out = []
    for i in range(N):
        own = {d for lst in pool_lists for d in lst[i]}
        while True:
            pick = rng.sample(pool, 3)
            if not own.intersection(pick):
                break
        out.append(pick)
    return out


def run_cells(cells, out_path, runs=3, workers=8):
    """cells: {name: (questions[list], docs[list])}"""
    out = L(out_path) if os.path.exists(out_path) else {}
    for cell, (qs, docs) in cells.items():
        if cell in out:
            acc = sum(r["pred"] == r["correct_answer"] for r in out[cell]) / N
            print(f"[skip] {cell}: {acc:.3f}", flush=True)
            continue
        votes = [[] for _ in range(N)]
        for run in range(runs):
            with ThreadPoolExecutor(workers) as ex:
                preds = list(ex.map(lambda i: answer(qs[i], QUESTIONS[i]["option"], docs[i]), range(N)))
            for i, p in enumerate(preds):
                votes[i].append(p)
            print(f"{cell} run {run+1}: {sum(p == q['answer'] for p, q in zip(preds, QUESTIONS))/N:.3f}", flush=True)
        out[cell] = [{"id": q["id"], "correct_answer": q["answer"], "votes": votes[i],
                      "pred": Counter(votes[i]).most_common(1)[0][0]} for i, q in enumerate(QUESTIONS)]
        acc = sum(r["pred"] == r["correct_answer"] for r in out[cell]) / N
        print(f"=== {cell} majority: {acc:.3f} ===", flush=True)
        json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out
