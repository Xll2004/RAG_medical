# exp1b_grounding_judge.py
# 新版 Experiment 1：生成时的解释是否依据了检索到的段落？（替代旧的“事后追问”版本）
#
# 数据：原始流水线 run_condition{1..5}.py 里 GPT-4o-mini 生成的解释（"answer" 字段）
#       生成 prompt："...Based on the following context... Provide a brief explanation and indicate which option is correct."
#       注意：这些解释用的是英文译文选项、默认 temperature、max_tokens=300，与论文的单字母协议不同
# 判别：LLM judge 读 段落（编号）+ 生成器看到的题干和选项 + 解释，输出
#   chosen_option  解释最终选的选项（A-E / none）
#   grounding      evidence | generic | none
#   claims_support 解释是否声称/暗示检索段落支持它选的答案
#   span           若 grounding=evidence：从段落中逐字复制的、解释所依据的一段话（≤30 词）
# 核验：span 必须能在所注明的段落里字符串匹配到，否则 evidence 降级为 unverified
# 之后只保留 judge 抽出的选项 == 论文中该条件的预测字母（pred_cX）的题
#
# 用法：python exp1b_grounding_judge.py --provider claude   （第二个 judge：--provider gpt）
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor
from llm_client import LLM

ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["claude", "gpt"], required=True)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()
OUT = f"exp1b_grounding_{args.provider}.json"

L = lambda f: json.load(open(f, encoding="utf-8"))
COND = {  # 条件 -> (结果文件, 题干字段, 选项字段)
    "C1": ("results_condition1_200_clean.json", "question_en_baseline", "options_en"),
    "C2": ("results_condition2_200.json", "question_mt", "option_mt"),
    "C3": ("results_condition3_200.json", "question_corrected", "options_corrected"),
    "C3B": ("results_condition3b_200.json", "question_corrected", "options_corrected"),
    "C4": ("results_condition4_200.json", "question_corrected", "options_corrected"),
    "C5": ("results_condition5_200.json", "question_corrected", "options_corrected"),
}

SYSTEM = "You are a careful annotator. Follow the output format exactly."
PROMPT = """A question-answering system was given three retrieved passages, a multiple-choice question, and its options, and wrote the explanation below. Your job is to judge how the explanation uses the passages. Do not judge whether the answer is medically correct.

Passages:
{passages}

Question: {question}

Options:
{options}

Explanation written by the system:
\"\"\"{explanation}\"\"\"

Answer four things:
1. chosen_option: the option letter (A-E) the explanation finally selects as correct, or "none" if it selects no single option (for example, it is cut off before choosing).
2. grounding:
   - "evidence": the explanation uses specific information from at least one passage (a finding, fact, term or statement that appears in the passage) as a reason for its choice;
   - "generic": the explanation mentions the passages or "the context" only in general terms, or says they are not relevant, and reasons from general knowledge;
   - "none": the explanation does not refer to the passages at all.
3. claims_support: true if the explanation states or implies that the passages support its chosen option, otherwise false.
4. If grounding is "evidence": the number of the passage it relies on most, and a span of at most 30 words copied verbatim from that passage that the explanation relies on. Otherwise use 0 and an empty span.

Reply with JSON only:
{{"chosen_option": "<A-E or none>", "grounding": "<evidence|generic|none>", "claims_support": <true|false>, "passage": <0-3>, "span": "<verbatim text or empty>"}}"""

norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower().strip(" .\"'")


def verify(span, passage):
    s, p = norm(span), norm(passage)
    if len(s) < 8:
        return False
    if s in p:
        return True
    parts = [x.strip() for x in re.split(r"\.\.\.|…", s) if len(x.strip()) >= 8]
    return bool(parts) and all(x in p for x in parts)


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


llm = LLM(args.provider)
out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
for cond, (fn, qk, ok) in COND.items():
    if cond in out and len(out[cond]) == 200:
        print(f"[skip] {cond}")
        continue
    rows = L(fn)

    def run(r):
        docs = r["retrieved_docs"]
        rec = dict(id=r["id"], correct_answer=r["correct_answer"], excluded=bool(r.get("cleaned")))
        opts = r.get(ok) or {}
        prompt = PROMPT.format(
            passages="\n\n".join(f"[{k+1}] {d}" for k, d in enumerate(docs)),
            question=r[qk], options="\n".join(f"{k}: {v}" for k, v in opts.items()),
            explanation=r["answer"])
        raw = llm.chat([{"role": "user", "content": prompt}], system=SYSTEM, max_tokens=400)
        j = parse(raw)
        g = str(j.get("grounding", "")).lower()
        p = int(j.get("passage", 0) or 0) if str(j.get("passage", "0")).isdigit() else 0
        span = str(j.get("span", ""))
        ver = g == "evidence" and 1 <= p <= len(docs) and verify(span, docs[p - 1])
        return {**rec, "chosen_option": str(j.get("chosen_option", "none")).strip().upper()[:4],
                "grounding": g, "claims_support": bool(j.get("claims_support", False)),
                "passage": p, "span": span, "span_verified": ver, "raw": raw}

    with ThreadPoolExecutor(args.workers) as ex:
        out[cond] = list(ex.map(run, rows))
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print(cond, Counter(x["grounding"] for x in out[cond]), "verified", sum(x["span_verified"] for x in out[cond]))
