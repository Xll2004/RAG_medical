# exp1_attribution.py
# Experiment 1：生成器答错时，会不会指着某一段检索到的段落当作依据？
#
# 设计（事后归因，post-hoc attribution）：
#   - 每个格子里生成器的答案保持不变（直接用论文里已有的多数投票 pred）
#   - 第一轮对话 = 原实验的 prompt（逐字一致）+ 生成器原来的答案
#   - 第二轮追问：你的答案依据的是哪一段？给出段落编号（0 = 没有依据任何段落）并逐字引用一句
#   - 引用句是否真的出自该段落，用字符串匹配核验（不靠 LLM 判断）
#
# 用法：
#   python exp1_attribution.py --provider claude   # Claude Sonnet 4.6，4 个格子
#   python exp1_attribution.py --provider gpt      # GPT-4o-mini，6 个格子（含原始中文题干）
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor
from llm_client import LLM

ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["claude", "gpt"], required=True)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--neutral", action="store_true", help="追问中去掉 \"own medical knowledge\" 的例子，检查措辞是否影响结果")
ap.add_argument("--cells", nargs="*", default=None)
args = ap.parse_args()
OUT = f"exp1_attribution_{args.provider}{'_neutral' if args.neutral else ''}.json"

L = lambda f: json.load(open(f, encoding="utf-8"))
gold = L("gold_set_200_translated.json")
c1 = L("results_condition1_200_clean.json")
c2 = L("results_condition2_200.json")
n = len(gold)
assert [r["id"] for r in c1] == [q["id"] for q in gold] == [r["id"] for r in c2]

Q = {"c1q": lambda i: c1[i]["question_en_baseline"],
     "c2q": lambda i: c2[i]["question_mt"],
     "zhq": lambda i: gold[i]["question"]}
D = {"c1docs": lambda i: c1[i]["retrieved_docs"],
     "c2docs": lambda i: c2[i]["retrieved_docs"]}

# ---------- 每个格子的已有答案 ----------
if args.provider == "claude":
    cl = L("results_claude_2x2_200.json")
    CELLS = {c: {r["id"]: r["pred"] for r in cl[f"claude_{c}"]}
             for c in ["c1q_c1docs", "c1q_c2docs", "c2q_c1docs", "c2q_c2docs"]}
    SYSTEM1 = ("Respond with only the single letter (A, B, C, D, or E) of the option you choose. "
               "Do not explain your reasoning.")   # 与 claude_2x2.py 相同
else:
    main = L("accuracy_results_200_clean.json")
    mc = L("results_matched_context_200.json")
    CELLS = {"c1q_c1docs": {r["id"]: r["pred_c1"] for r in main},     # 论文中 C1/C1 = 主实验 C1
             "c2q_c2docs": {r["id"]: r["pred_c2"] for r in main}}     # 论文中 C2/C2 = 主实验 C2
    for c in ["c1q_c2docs", "c2q_c1docs", "zhq_c1docs", "zhq_c2docs"]:
        CELLS[c] = {r["id"]: r["pred"] for r in mc[c]}
    SYSTEM1 = None


def original_prompt(question, options, docs):
    """与 run_accuracy.py 的 get_answer_choice 逐字一致"""
    context = "\n\n".join(docs)
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"""You are a medical expert. Based on the context below, select the correct answer.
Only reply with a single letter (A, B, C, D, or E).

Context:
{context}

Question: {question}

Options:
{options_text}

Answer:"""


def followup(docs, letter):
    numbered = "\n\n".join(f"[{k+1}] {d}" for k, d in enumerate(docs))
    return f"""Here are the context passages again, numbered in the order they appeared:

{numbered}

Which passage, if any, did your answer ({letter}) rely on? If it relied on one of the passages, give its number and copy the single most relevant sentence from that passage verbatim. {"If it did not rely on any passage, use 0 and an empty quote." if args.neutral else "If it did not rely on any passage (for example, because you used your own medical knowledge), use 0 and an empty quote."}

Reply with JSON only: {{"passage": <0, 1, 2 or 3>, "quote": "<verbatim sentence or empty string>"}}"""


SYSTEM2 = "Follow the format requested in the user's last message exactly."
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower().strip(" .\"'")


def verify(quote, passage):
    q, p = norm(quote), norm(passage)
    if len(q) < 10:
        return False
    if q in p:
        return True
    q2 = q.replace("...", " ").replace("…", " ")      # 允许省略号截断
    parts = [x.strip() for x in q2.split("  ") if len(x.strip()) >= 10] or [q2.strip()]
    return all(x in p for x in parts)


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0))
        return int(d.get("passage", -1)), str(d.get("quote", ""))
    except Exception:
        m2 = re.search(r'"passage"\s*:\s*(\d)', text)
        return (int(m2.group(1)) if m2 else -1), ""


if args.cells:
    CELLS = {c: CELLS[c] for c in args.cells}
llm = LLM(args.provider)
out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}

for cell, preds in CELLS.items():
    if cell in out and len(out[cell]) == n:
        print(f"[skip] {cell}")
        continue
    qk, dk = cell.split("_")
    def run(i):
        q = gold[i]
        pred = preds[q["id"]]
        docs = D[dk](i)
        rec = dict(id=q["id"], pred=pred, correct_answer=q["answer"], correct=pred == q["answer"])
        if pred not in "ABCDE" or len(pred) != 1:
            return {**rec, "passage": None, "quote": "", "verified": False, "raw": "unparsed pred"}
        msgs = [{"role": "user", "content": original_prompt(Q[qk](i), q["option"], docs)},
                {"role": "assistant", "content": pred},
                {"role": "user", "content": followup(docs, pred)}]
        raw = llm.chat(msgs, system=SYSTEM2, max_tokens=300)
        p, quote = parse(raw)
        ok = 1 <= p <= len(docs) and verify(quote, docs[p - 1])
        return {**rec, "passage": p, "quote": quote, "verified": ok, "raw": raw}
    with ThreadPoolExecutor(args.workers) as ex:
        out[cell] = list(ex.map(run, range(n)))
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    w = [r for r in out[cell] if not r["correct"]]
    cite = sum((r["passage"] or 0) > 0 for r in w)
    ver = sum(r["verified"] for r in w)
    print(f"{cell}: wrong={len(w)}  cite a passage={cite} ({cite/len(w):.1%})  verified quote={ver} ({ver/len(w):.1%})")
