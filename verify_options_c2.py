# verify_options_c2.py — 判断 6 月 3 日的 pred_c2 是用中文选项还是 OPUS 翻译选项打分的
# C2 问题 + C2 文档，分别配 option（中文）和 option_mt（OPUS）各跑 1 遍，与存档 pred_c2 逐题比对
import ast, json, os
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
c2 = json.load(open("results_condition2_200.json", encoding="utf-8"))
stored = {r["id"]: r["pred_c2"] for r in json.load(open("accuracy_results_200.json"))}
assert [x["id"] for x in c2[:200]] == [q["id"] for q in gold]

def job(args):
    i, key = args
    return get_answer_choice(c2[i]["question_mt"], gold[i][key], c2[i]["retrieved_docs"])

out = {}
with ThreadPoolExecutor(8) as ex:
    for key in ["option", "option_mt"]:
        preds = list(ex.map(job, [(i, key) for i in range(200)]))
        out[key] = preds
        agree = sum(p == stored[gold[i]["id"]] for i, p in enumerate(preds))
        acc = sum(p == gold[i]["answer"] for i, p in enumerate(preds)) / 200
        print(f"{key:10s}: accuracy {acc:.3f} | agree with stored pred_c2 {agree}/200")

json.dump(out, open("verify_options_c2_out.json", "w"), indent=1)
