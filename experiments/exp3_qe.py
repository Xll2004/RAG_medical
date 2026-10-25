# exp3_qe.py
# Experiment 3 (step 1): give every translated retrieval query a reference-free translation quality score (QE)
# Scoring prompt = reference-free GEMBA-DA (Kocmi & Federmann, 2023), used verbatim, 0-100
# Usage: python experiments/exp3_qe.py --provider claude    (or --provider gpt)
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor
from llm_client import LLM

ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["claude", "gpt"], required=True)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()
OUT = f"results/exp3_qe_{args.provider}.json"

L = lambda f: json.load(open(f, encoding="utf-8"))
gold = L("data/gold_set_200_translated.json")
SRC = {q["id"]: q["question"] for q in gold}
TGT = {
    "C1":  {r["id"]: r["question_en_baseline"] for r in L("results/results_condition1_200_clean.json")},
    "C1P": {r["id"]: r["question_en_gpt41"] for r in L("results/results_c1prime_200_clean.json")},
    "C2":  {r["id"]: r["question_mt"] for r in L("results/results_condition2_200.json")},
    "C3":  {r["id"]: r["question_corrected"] for r in L("results/results_condition3_200.json")},
    "C3B": {r["id"]: r["question_corrected"] for r in L("results/results_condition3b_200.json")},
    "C4":  {r["id"]: r["question_corrected"] for r in L("results/results_condition4_200.json")},
    "C5":  {r["id"]: r["question_corrected"] for r in L("results/results_condition5_200.json")},
}

GEMBA = ('Score the following translation from Chinese to English on a continuous scale from 0 to 100, '
         'where score of zero means "no meaning preserved" and score of one hundred means '
         '"perfect meaning and grammar".\n\nChinese source: "{src}"\nEnglish translation: "{tgt}"\nScore: ')
SYSTEM = "Reply with only the score, a number from 0 to 100."

llm = LLM(args.provider)
out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
for cond, tg in TGT.items():
    if cond in out and len(out[cond]) == len(gold):
        print(f"[skip] {cond}")
        continue
    def run(q):
        msgs = [{"role": "user", "content": GEMBA.format(src=SRC[q["id"]], tgt=tg[q["id"]])}]
        raw = llm.chat(msgs, system=SYSTEM, max_tokens=10)
        m = re.match(r"\s*\**(\d+(?:\.\d+)?)", raw)
        if m:
            return dict(id=q["id"], score=float(m.group(1)), raw=raw, fallback=False)
        # Reply starts with analysis: re-send the same request with a higher token limit and take the last score given (same idea as the fallback in claude_2x2.py)
        raw = llm.chat(msgs, system=SYSTEM, max_tokens=1024)
        ms = re.findall(r"(?i)score\W{0,5}(\d+(?:\.\d+)?)", raw) or re.findall(r"\b(\d+(?:\.\d+)?)\b", raw)
        return dict(id=q["id"], score=float(ms[-1]) if ms else None, raw=raw, fallback=True)
    with ThreadPoolExecutor(args.workers) as ex:
        out[cond] = list(ex.map(run, gold))
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    s = [r["score"] for r in out[cond] if r["score"] is not None]
    print(f"{cond}: mean QE = {sum(s)/len(s):.1f}  (unparsed {len(gold)-len(s)})")
