#!/bin/bash
# Run in the local rag-medical environment: conda activate rag-medical && bash experiments/run_gpt_new_experiments.sh
# Repeat the three new experiments with GPT-4o-mini (the Claude versions are done; results are in *_claude.json)
set -e
python experiments/exp1_attribution.py --provider gpt
python experiments/exp1_attribution.py --provider gpt --neutral --cells c2q_c2docs c2q_c1docs zhq_c2docs
python experiments/exp3_qe.py --provider gpt
python experiments/exp_memorization.py --provider gpt
python analysis/analyze_new_experiments.py --provider gpt | tee logs/analysis_gpt.log
