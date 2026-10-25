# verify_options_all.py — 对每个条件判断：存档预测是用中文选项还是翻译/纠正后的选项打分的
# 每个条件：同样的问题 + 同样的检索文档，分别配 (a) 中文 option (b) 该条件生成的英文选项，各跑 1 遍，
# 与存档的预测逐题比对。哪种选项的一致率接近 ~97%（即重复运行的波动水平），存档就是用哪种跑的。
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

get_answer_choice = load_func("run_accuracy.py", "get_answer_choice")
gold = json.load(open("gold_set_200_translated.json", encoding="utf-8"))
ids = [q["id"] for q in gold]

def load(fn): return json.load(open(fn, encoding="utf-8"))

c1 = load("results_condition1_200.json")   # 原始（未清理）C1，对应 6/3 的 pred_c1
c2 = load("results_condition2_200.json")

# (名称, 结果列表, 问题字段, 英文选项来源 lambda i, 存档文件, 存档字段)
CONDS = [
    ("C1",  c1, "question_en_baseline", lambda i: c1[i]["options_en"],       "accuracy_results_200.json",     "pred_c1"),
    ("C2",  c2, "question_mt",          lambda i: gold[i]["option_mt"],      "accuracy_results_200.json",     "pred_c2"),
]
for name, fn, pk in [("C3", "results_condition3_200.json", "pred_c3"),
                     ("C3b", "results_condition3b_200.json", "pred_c3b"),
                     ("C4", "results_condition4_200.json", "pred_c4"),
                     ("C5", "results_condition5_200.json", "pred_c5")]:
    r = load(fn)
    acc_fn = "accuracy_results_c3b_200.json" if name == "C3b" else "accuracy_results_200.json"
    CONDS.append((name, r, "question_corrected", (lambda r: lambda i: r[i]["options_corrected"])(r), acc_fn, pk))
c1p = load("results_c1prime_200.json")
CONDS.append(("C1prime", c1p, "question_en_gpt41", lambda i: c1p[i]["options_en"], "accuracy_c1prime_200.json", "pred_c1prime"))

def call(q, opts, docs):
    for t in range(5):
        try:
            return get_answer_choice(q, opts, docs)
        except Exception:
            time.sleep(2 * (t + 1))
    return "ERR"

OUT = "verify_options_all_out.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
print(f"{'cond':8s} {'options':10s} {'acc':>6s} {'stored acc':>10s} {'agree':>8s}")
for name, res, qkey, en_opts, acc_fn, pk in CONDS:
    assert [x["id"] for x in res[:200]] == ids, f"{name} id 未对齐"
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
