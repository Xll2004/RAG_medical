# m4_agent_loop.py -- reviewer comment M4: a minimal closed-loop agent
# Default action: OPUS-MT translation (C2) -> retrieval -> answer. Before answering, the agent runs one check; if it fails, it "re-translates (LLM, from the Chinese original) + re-retrieves",
# i.e. upgrades to the C1 path. Two check signals are compared:
#   (i)  translation quality score (GEMBA, existing exp3_qe_*.json, available before retrieval)
#   (ii) evidence-relevance self-check: GPT-4o-mini reads the Chinese question + the 3 retrieved passages and judges whether the evidence is relevant (after retrieval, before answering)
# This script only makes the API calls needed for (ii) (200 each for C2 and C1 evidence; C1 is used to estimate the false-alarm rate); the policy simulation is in analyze_review2.py
# Usage: python -u experiments/m4_agent_loop.py | tee logs/m4.log
from review2_common import *

OUT = "results/m4_relevance_gpt.json"
PROMPT = """You are checking retrieved evidence before answering a medical question.

Question (original language):
{q}

Retrieved passages:
{p}

Do these passages contain information that is relevant to answering this question? Reply with only "yes" or "no"."""
out = L(OUT) if os.path.exists(OUT) else {}
for key, docs in [("c2docs", C2DOCS), ("c1docs", C1DOCS)]:
    if key in out:
        print(f"[skip] {key}")
        continue
    def judge(i):
        p = "\n\n".join(f"[{k+1}] {d}" for k, d in enumerate(docs[i]))
        return chat(PROMPT.format(q=ZHQ[i], p=p), max_tokens=3).lower()
    with ThreadPoolExecutor(8) as ex:
        out[key] = [{"id": q["id"], "verdict": v} for q, v in zip(QUESTIONS, ex.map(judge, range(N)))]
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{key}: judged relevant {sum(r['verdict'].startswith('yes') for r in out[key])}/{N}", flush=True)
print("M4 done")
