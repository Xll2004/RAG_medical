#!/bin/bash
# 在本地 rag-medical 环境里运行：conda activate rag-medical && bash run_gpt_new_experiments.sh
# 用 GPT-4o-mini 重复三个新实验（Claude 版已经跑完，结果在 *_claude.json）
set -e
python exp1_attribution.py --provider gpt
python exp1_attribution.py --provider gpt --neutral --cells c2q_c2docs c2q_c1docs zhq_c2docs
python exp3_qe.py --provider gpt
python exp_memorization.py --provider gpt
python analyze_new_experiments.py --provider gpt | tee analysis_gpt.log
