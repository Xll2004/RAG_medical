# claude_2x2.py
# Reproduce the key matched-context cells (question x document type) with Claude Sonnet
# User prompt, options and voting are identical to the main experiment: the generation function is taken verbatim from the original script; only the client is swapped for a Claude adapter
# Only two differences from GPT, both to be stated in the appendix:
#   1) the system message restricts output to a single letter
#   2) if the reply starts with an explanation and no letter can be extracted, the same request is re-sent with a higher token limit and the final answer is taken (fallback)
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

# ---------- Answer parsing (no effect on single-letter replies) ----------
ANS_PATTERNS = [
    re.compile(r"(?i:answer)\s*(?:(?i:is))?\s*[:：]?\s*\**\s*\(?([A-E])\b"),   # "The answer is B" / "Answer: **C**"
    re.compile(r"^\s*\**\(?([A-E])\)?(?:[\s\.\):*]|$)"),                        # Starts with a letter: "B" / "B." / "**B**"
    re.compile(r"\*\*\(?([A-E])\)?[\.\):]?\*\*"),                               # Bold letter "**D**"
]
LINE_LETTER = re.compile(r"(?m)^\s*\**\(?([A-E])\)?\.?\**\s*$")               # Letter on its own line

def extract_letter(text):
    """Short reply: take the first match"""
    for p in ANS_PATTERNS:
        m = p.search(text)
        if m:
            return m.group(1)
    return None

def extract_last(text):
    """Full reply: take the last answer mentioned, to avoid letters mentioned mid-reasoning"""
    for p in ANS_PATTERNS + [LINE_LETTER]:
        ms = list(p.finditer(text))
        if ms:
            return ms[-1].group(1)
    return None

UNPARSED = []   # Replies still unparseable after fallback
FALLBACK = []   # Replies that triggered fallback (tail of the full reply)

# ---------- Claude call (with retries) ----------
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

# ---------- Adapter: forward client.chat.completions.create(...) to Claude ----------
class _Completions:
    def create(self, model, messages, max_tokens, temperature=0, **kw):
        text = _call(messages, max(max_tokens, 10), temperature)
        letter = extract_letter(text)
        if letter is None:
            # Fallback: same request with a higher token limit so the model answers fully; take the final answer
            text2 = _call(messages, 1024, temperature)
            letter = extract_last(text2)
            FALLBACK.append(text2[-300:])
            if letter is None:
                UNPARSED.append(text2[-300:])
                letter = text2       # If still unparseable, return as-is; the original function takes the first character, which is counted as invalid
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=letter))])

client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

def load_func(path, name):
    """Extract only the function definitions from the original script (without running the rest), so the prompt is verbatim identical to the original experiment"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"client": client}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns[name]

get_answer_choice = load_func("experiments/run_accuracy.py", "get_answer_choice")
get_answer_choice_no_context = load_func("experiments/run_norag_baseline.py", "get_answer_choice_no_context")

# ---------- Data ----------
questions = json.load(open("data/gold_set_200_translated.json", encoding="utf-8"))
c1 = json.load(open("results/results_condition1_200_clean.json", encoding="utf-8"))
c2 = json.load(open("results/results_condition2_200.json", encoding="utf-8"))
n = len(questions)
assert [x["id"] for x in c1[:n]] == [q["id"] for q in questions], "C1 ids not aligned"
assert [x["id"] for x in c2[:n]] == [q["id"] for q in questions], "C2 ids not aligned"

C1Q = lambda i: c1[i]["question_en_baseline"]
C2Q = lambda i: c2[i]["question_mt"]
C1D = lambda i: c1[i]["retrieved_docs"]
C2D = lambda i: c2[i]["retrieved_docs"]

# ---------- Cells: question {C1, C2} x docs {none, C1 docs, C2 docs} ----------
CELLS = {
    "claude_c1q_none":   (C1Q, None),
    "claude_c1q_c1docs": (C1Q, C1D),   # = Claude version of C1 (replaces the original C1b)
    "claude_c2q_none":   (C2Q, None),
    "claude_c2q_c2docs": (C2Q, C2D),   # = Claude version of C2
    "claude_c2q_c1docs": (C2Q, C1D),   # MT question + correct evidence
    "claude_c1q_c2docs": (C1Q, C2D),   # Correct question + wrong evidence
}

# ---------- Run (3 runs + majority vote, saved per cell, resumable) ----------
OUT = "results/results_claude_2x2_200.json"
out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
N_RUNS = 3

for cell, (qf, cf) in CELLS.items():
    if cell in out:
        print(f"[skip] {cell} already done")
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
            print(f"  !! too many invalid replies, examples: {raw_invalid[:5]}")
            print("  !! tails of replies still unparseable after fallback:")
            for t in UNPARSED[:3]:
                print("     ", repr(t))
            print("  !! stopping; check Claude's output format first")
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