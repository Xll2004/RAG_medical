# matched_context_ablation.py
import ast, json, os, random, time
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_func(path, name):
    """只取出原脚本中的函数定义（不执行其余代码），保证 prompt 与原实验逐字一致"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"client": client}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns[name]

get_answer_choice = load_func("run_accuracy.py", "get_answer_choice")
get_answer_choice_no_context = load_func("run_norag_baseline.py", "get_answer_choice_no_context")

# ---------- 数据 ----------
questions = json.load(open("gold_set_200_translated.json", encoding="utf-8"))
res = {c: json.load(open(f"results_condition{c}_200.json", encoding="utf-8")) for c in [2, 3, 5]}
res[1] = json.load(open("results_condition1_200_clean.json", encoding="utf-8"))
n = len(questions)
for c, d in res.items():
    assert [x["id"] for x in d[:n]] == [q["id"] for q in questions], f"C{c} id 未对齐"

# ---------- random docs：固定种子，提前生成 ----------
rng = random.Random(42)
pool = sorted({doc for d in res.values() for item in d[:n] for doc in item["retrieved_docs"]})
random_ctx = []
for i in range(n):
    own = {doc for d in res.values() for doc in d[i]["retrieved_docs"]}
    while True:
        pick = rng.sample(pool, 3)
        if not own.intersection(pick):
            break
    random_ctx.append(pick)
print(f"random pool size: {len(pool)}")

# ---------- 格子定义 ----------
C1Q = lambda i: res[1][i]["question_en_baseline"]
C2Q = lambda i: res[2][i]["question_mt"]
C3Q = lambda i: res[3][i]["question_corrected"]
C5Q = lambda i: res[5][i]["question_corrected"]
CELLS = {
    "c1q_none":   (C1Q, None),
    "c1q_c1docs": (C1Q, lambda i: res[1][i]["retrieved_docs"]),   # sanity check
    "c1q_c2docs": (C1Q, lambda i: res[2][i]["retrieved_docs"]),
    "c1q_c3docs": (C1Q, lambda i: res[3][i]["retrieved_docs"]),
    "c1q_c5docs": (C1Q, lambda i: res[5][i]["retrieved_docs"]),
    "c1q_random": (C1Q, lambda i: random_ctx[i]),
    "c2q_none":   (C2Q, None),
    "c3q_c2docs": (C3Q, lambda i: res[2][i]["retrieved_docs"]),
    "c5q_c2docs": (C5Q, lambda i: res[2][i]["retrieved_docs"]),
    "c2q_c1docs": (C2Q, lambda i: res[1][i]["retrieved_docs"]),
    "c2q_random": (C2Q, lambda i: random_ctx[i]),
    "zhq_none":   (lambda i: questions[i]["question"], None),
    "zhq_c1docs": (lambda i: questions[i]["question"], lambda i: res[1][i]["retrieved_docs"]),
    "zhq_c2docs": (lambda i: questions[i]["question"], lambda i: res[2][i]["retrieved_docs"]),
    "zhq_random": (lambda i: questions[i]["question"], lambda i: random_ctx[i]),
}

# ---------- 运行（3 runs + 多数投票，与 run_accuracy.py 相同顺序）----------
OUT = "results_matched_context_200.json"
out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
N_RUNS = 3

for cell, (qf, cf) in CELLS.items():
    if cell in out:
        print(f"[skip] {cell} 已完成")
        continue
    votes = [[] for _ in range(n)]
    for run in range(N_RUNS):
        correct = 0
        for i, q in enumerate(questions):
            if cf is None:
                pred = get_answer_choice_no_context(qf(i), q["option"])
            else:
                pred = get_answer_choice(qf(i), q["option"], cf(i))
            votes[i].append(pred)
            correct += int(pred == q["answer"])
            time.sleep(0.2)
        print(f"{cell} run {run+1}: {correct/n:.3f}")
    out[cell] = [{
        "id": q["id"],
        "correct_answer": q["answer"],
        "votes": votes[i],
        "pred": Counter(votes[i]).most_common(1)[0][0],
    } for i, q in enumerate(questions)]
    acc = sum(r["pred"] == r["correct_answer"] for r in out[cell]) / n
    print(f"=== {cell} majority: {acc:.3f} ===")
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---------- sanity check：c1q_c1docs vs 原 pred_c1 ----------
orig = {r["id"]: r["pred_c1"] for r in json.load(open("accuracy_results_200_clean.json"))}
agree = sum(r["pred"] == orig[r["id"]] for r in out["c1q_c1docs"])
print(f"\nsanity: c1q_c1docs 与原 pred_c1 一致 {agree}/{n}")