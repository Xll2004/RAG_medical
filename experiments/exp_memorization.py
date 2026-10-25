# exp_memorization.py
# Memorisation check: CMB-Exam questions are public, so the generator may have seen them in training
#   (a) Ask directly: have you seen this question? Which exam is it from? (self-report, reference only)
#   (b) Option completion: give the question + options A-D (original Chinese) and ask the model to write option E verbatim;
#       getting it verbatim right suggests the question may be memorised (guided completion in the style of Golchin & Surdeanu 2024)
# Usage: python experiments/exp_memorization.py --provider claude   (or --provider gpt)
import argparse, json, re
from concurrent.futures import ThreadPoolExecutor
from llm_client import LLM

ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["claude", "gpt"], required=True)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()
OUT = f"results/exp_memorization_{args.provider}.json"

gold = json.load(open("data/gold_set_200_translated.json", encoding="utf-8"))
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

# Strip whitespace and punctuation, including full-width Chinese punctuation, before comparing
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
