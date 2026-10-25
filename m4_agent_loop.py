# m4_agent_loop.py —— 审稿意见 M4：最小闭环 agent
# 默认动作：OPUS-MT 翻译（C2）-> 检索 -> 作答。agent 在作答前做一次检查，未通过就"重新翻译（LLM，从中文原文）+ 重新检索"，
# 即升级为 C1 路径。比较两种检查信号：
#   (i)  翻译质量分（GEMBA，已有 exp3_qe_*.json，检索前即可得到）
#   (ii) 证据相关性自检：GPT-4o-mini 读中文原题 + 检索到的 3 段证据，判断证据是否与原题相关（检索后、作答前）
# 本脚本只跑 (ii) 需要的 API 调用（C2 和 C1 证据各 200 次，C1 用来估计误报率）；策略模拟在 analyze_review2.py 里做
# 用法：python -u m4_agent_loop.py | tee m4.log
from review2_common import *

OUT = "m4_relevance_gpt.json"
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
