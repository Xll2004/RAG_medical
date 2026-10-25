# claude_2x2.py
# 用 Claude Sonnet 复现 matched-context 的关键格子（题干 × 文档类型）
# user prompt、选项、投票方式与主实验完全一致：生成函数从原脚本中原样取出，只把 client 换成 Claude 适配器
# 与 GPT 的差异仅有两处，均需在 appendix 说明：
#   1) system message 限定只输出一个字母
#   2) 回复以解释开头、抽不出字母时，用同一请求放宽 token 上限重新作答，取最终答案（fallback）
import ast, json, os, re, time
from collections import Counter
from types import SimpleNamespace
from dotenv import load_dotenv
import anthropic

load_dotenv()
ac = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM = ("Respond with only the single letter (A, B, C, D, or E) of the option you choose. "
          "Do not explain your reasoning.")

# ---------- 答案解析（对单字母回复不产生任何影响）----------
ANS_PATTERNS = [
    re.compile(r"(?i:answer)\s*(?:(?i:is))?\s*[:：]?\s*\**\s*\(?([A-E])\b"),   # "The answer is B" / "Answer: **C**"
    re.compile(r"^\s*\**\(?([A-E])\)?(?:[\s\.\):*]|$)"),                        # 以字母开头："B" / "B." / "**B**"
    re.compile(r"\*\*\(?([A-E])\)?[\.\):]?\*\*"),                               # 加粗的字母 "**D**"
]
LINE_LETTER = re.compile(r"(?m)^\s*\**\(?([A-E])\)?\.?\**\s*$")               # 单独成行的字母

def extract_letter(text):
    """短回复：取第一个匹配"""
    for p in ANS_PATTERNS:
        m = p.search(text)
        if m:
            return m.group(1)
    return None

def extract_last(text):
    """完整回复：取最后一次出现的答案，避免抓到推理中途提到的字母"""
    for p in ANS_PATTERNS + [LINE_LETTER]:
        ms = list(p.finditer(text))
        if ms:
            return ms[-1].group(1)
    return None

UNPARSED = []   # fallback 后仍抽不出字母的回复
FALLBACK = []   # 触发 fallback 的回复（完整回复的结尾部分）

# ---------- Claude 调用（带重试）----------
def _call(messages, max_tokens, temperature):
    for attempt in range(5):
        try:
            r = ac.messages.create(model=CLAUDE_MODEL, system=SYSTEM, messages=messages,
                                   max_tokens=max_tokens, temperature=temperature)
            return r.content[0].text if r.content else ""
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.InternalServerError) as e:
            wait = 2 ** attempt * 5
            print(f"  API error ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Claude API failed after 5 retries")

# ---------- 适配器：把 client.chat.completions.create(...) 转发到 Claude ----------
class _Completions:
    def create(self, model, messages, max_tokens, temperature=0, **kw):
        text = _call(messages, max(max_tokens, 10), temperature)
        letter = extract_letter(text)
        if letter is None:
            # fallback：同一请求放宽 token 上限，让模型完整作答，取最终答案
            text2 = _call(messages, 1024, temperature)
            letter = extract_last(text2)
            FALLBACK.append(text2[-300:])
            if letter is None:
                UNPARSED.append(text2[-300:])
                letter = text2       # 仍抽不出时原样返回，原函数取首字符后会被计为 invalid
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=letter))])

client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

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
c1 = json.load(open("results_condition1_200_clean.json", encoding="utf-8"))
c2 = json.load(open("results_condition2_200.json", encoding="utf-8"))
n = len(questions)
assert [x["id"] for x in c1[:n]] == [q["id"] for q in questions], "C1 id 未对齐"
assert [x["id"] for x in c2[:n]] == [q["id"] for q in questions], "C2 id 未对齐"

C1Q = lambda i: c1[i]["question_en_baseline"]
C2Q = lambda i: c2[i]["question_mt"]
C1D = lambda i: c1[i]["retrieved_docs"]
C2D = lambda i: c2[i]["retrieved_docs"]

# ---------- 格子：题干 {C1, C2} × 文档 {无, C1 docs, C2 docs} ----------
CELLS = {
    "claude_c1q_none":   (C1Q, None),
    "claude_c1q_c1docs": (C1Q, C1D),   # = Claude 版 C1（替代原 C1b）
    "claude_c2q_none":   (C2Q, None),
    "claude_c2q_c2docs": (C2Q, C2D),   # = Claude 版 C2
    "claude_c2q_c1docs": (C2Q, C1D),   # MT 题干 + 正确证据
    "claude_c1q_c2docs": (C1Q, C2D),   # 正确题干 + 错误证据
}

# ---------- 运行（3 runs + 多数投票，逐格存盘，可断点续跑）----------
OUT = "results_claude_2x2_200.json"
out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
N_RUNS = 3

for cell, (qf, cf) in CELLS.items():
    if cell in out:
        print(f"[skip] {cell} 已完成")
        continue
    votes = [[] for _ in range(n)]
    raw_invalid = []
    fallback_counts = []
    for run in range(N_RUNS):
        correct = invalid = 0
        UNPARSED.clear()
        fb_before = len(FALLBACK)
        for i, q in enumerate(questions):
            if cf is None:
                pred = get_answer_choice_no_context(qf(i), q["option"])
            else:
                pred = get_answer_choice(qf(i), q["option"], cf(i))
            votes[i].append(pred)
            correct += int(pred == q["answer"])
            if pred not in ("A", "B", "C", "D", "E"):
                invalid += 1
                raw_invalid.append((run, q["id"], pred))
            time.sleep(0.2)
        fb = len(FALLBACK) - fb_before
        fallback_counts.append(fb)
        print(f"{cell} run {run+1}: acc {correct/n:.3f}  invalid {invalid}  fallback {fb}")
        if run == 0 and invalid > 5:
            print(f"  !! invalid 过多，示例: {raw_invalid[:5]}")
            print("  !! fallback 后仍无法解析的回复结尾:")
            for t in UNPARSED[:3]:
                print("     ", repr(t))
            print("  !! 停止运行，先检查 Claude 的输出格式")
            raise SystemExit(1)
    out[cell] = [{
        "id": q["id"],
        "correct_answer": q["answer"],
        "votes": votes[i],
        "pred": Counter(votes[i]).most_common(1)[0][0],
    } for i, q in enumerate(questions)]
    out.setdefault("_meta", {})[cell] = {"fallback_per_run": fallback_counts}
    acc = sum(r["pred"] == r["correct_answer"] for r in out[cell]) / n
    print(f"=== {cell} majority: {acc:.3f}  (fallback per run: {fallback_counts}) ===")
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\nall cells done:", [k for k in out if k != "_meta"])