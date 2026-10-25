# exp_memorization.py
# 记忆检查：CMB-Exam 题目是公开的，生成器可能在训练中见过
#   (a) 直接问：你见过这道题吗？出自哪个考试？（自我报告，只作参考）
#   (b) 选项补全：给题干 + 选项 A-D（原始中文），让模型逐字写出选项 E；
#       能逐字写对说明可能背过这道题（Golchin & Surdeanu 2024 风格的 guided completion）
# 用法：python exp_memorization.py --provider claude   （或 --provider gpt）
import argparse, json, re
from concurrent.futures import ThreadPoolExecutor
from llm_client import LLM

ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["claude", "gpt"], required=True)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()
OUT = f"exp_memorization_{args.provider}.json"

gold = json.load(open("gold_set_200_translated.json", encoding="utf-8"))
SYSTEM = "Follow the format requested by the user exactly."

ASK = """Below is a question from a Chinese medical examination, with its answer options.

Question: {q}
Options:
{opts}

Have you seen this exact question before, for example in your training data? Answer honestly; if you are not sure, say "unsure".
Reply with JSON only: {{"seen": "yes" | "no" | "unsure", "source": "<exam or dataset name if you recognise it, otherwise empty>"}}"""

COMPLETE = """Below is a question from a Chinese medical examination (CMB). Options A to D are shown; option E is hidden.

Question: {q}
Options:
{opts_ad}
E: ???

If you have seen this question before, write option E exactly as it appears in the original exam. If not, give your best guess of the original wording.
Reply with JSON only: {{"E": "<option E text>"}}"""

norm = lambda s: re.sub(r"[\s。．.、，,：:；;\"'“”（）()]", "", s or "")


def field(text, key):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return str(json.loads(m.group(0)).get(key, ""))
    except Exception:
        m2 = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        return m2.group(1) if m2 else ""


llm = LLM(args.provider)


def run(q):
    opts = q["option"]
    full = "\n".join(f"{k}: {v}" for k, v in opts.items())
    ad = "\n".join(f"{k}: {v}" for k, v in opts.items() if k != "E")
    r1 = llm.chat([{"role": "user", "content": ASK.format(q=q["question"], opts=full)}], system=SYSTEM, max_tokens=150)
    r2 = llm.chat([{"role": "user", "content": COMPLETE.format(q=q["question"], opts_ad=ad)}], system=SYSTEM, max_tokens=150)
    e_pred = field(r2, "E")
    return dict(id=q["id"], seen=field(r1, "seen").lower(), source=field(r1, "source"), raw_ask=r1,
                e_true=opts.get("E", ""), e_pred=e_pred,
                e_exact=bool(e_pred) and norm(e_pred) == norm(opts.get("E", "")), raw_complete=r2)


with ThreadPoolExecutor(args.workers) as ex:
    rows = list(ex.map(run, [q for q in gold if "E" in q["option"]]))
json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
print("self-report:", Counter(r["seen"] for r in rows))
print(f"option-E exact reproduction: {sum(r['e_exact'] for r in rows)}/{len(rows)}")
