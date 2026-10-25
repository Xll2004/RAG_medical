# Lost in the Question: Translation Errors and the Evidence They Retrieve in Cross-Lingual Medical RAG

Xuelin Liang, Andrew Ju

Code, prompts and per-question outputs for the paper accepted at **AMAI 2026** (First Workshop on Agentic Medical AI, co-located with IEEE BIBM 2026).

A medical agent that searches English evidence for a Chinese question must first translate the question into a search query. This repository isolates that single *translate-then-retrieve* step in a fixed RAG pipeline and varies only the translation, using 200 Chinese exam questions from CMB-Exam, a PubMed-derived corpus (PubMedQA, 852,618 passages) and two generators (GPT-4o-mini, Claude Sonnet 4.6).

Main findings:

- OPUS-MT retrieves evidence largely disjoint from a faithful LLM translation (mean top-3 Jaccard 0.135) and is **21.0 points** less accurate (55.0% vs 34.0%).
- A Shapley decomposition assigns most of the gap to the misread question (65%; Claude 84%) and the rest to *mistranslation-coherent* evidence, which harms mainly a generator that shares the misreading (interaction 10.0 points, 95% CI [3.0, 17.5]).
- A reference-free translation-quality score (GEMBA-DA) is associated with failure before the evidence is used (AUC 0.74), and answering the original question with the same retrieved evidence closes the gap without a new search.

## Repository layout

```
experiments/   run the pipeline and generate results (calls LLM APIs)
data_prep/     sample, translate and clean the 200 questions
annotation/    human error-taxonomy annotation and agreement
analysis/      statistical tests and analyses (read JSON only)
paper/         paper_numbers.py (every number in the paper), plot_figures.py, figures/
data/          question set, human annotations, cleaning logs
results/       per-question outputs for every condition, caches
logs/          console output of past runs
```

All scripts are run from the **repository root**, e.g. `python experiments/run_condition1.py`. In PyCharm, set the run configuration's working directory to the project root (`$ProjectFileDir$`).

## Experimental conditions

Each condition is a different way of producing the English search query; retrieval (all-MiniLM-L6-v2 + FAISS IndexFlatL2, top-3) and generation are otherwise fixed. The generator always sees the five answer options in the original Chinese.

| Condition | Translation action | Script | Output |
|---|---|---|---|
| C1 | GPT-4o-mini translation (high-quality condition) | `experiments/run_condition1.py` | `results/results_condition1_200.json` |
| C1' | GPT-4.1 translation | `experiments/run_c1prime.py` | `results/results_c1prime_200.json` |
| C2 | Helsinki-NLP `opus-mt-zh-en` | `data_prep/translate.py`, `experiments/run_pipeline.py` | `results/results_condition2_200.json` |
| C3 | C2 + GPT-4o-mini correction, re-retrieval | `experiments/run_condition3.py` | `results/results_condition3_200.json` |
| C3b | as C3, "only replace incorrect terms" | `experiments/run_condition3b.py` | `results/results_condition3b_200.json` |
| C4 | C2 + DeepSeek V4-Flash correction, re-retrieval | `experiments/run_condition4.py` | `results/results_condition4_200.json` |
| C5 | C2 + GPT-5.5 correction, re-retrieval | `experiments/run_condition5.py` | `results/results_condition5_200.json` |

Single-letter answers (temperature 0, three runs, majority vote) are scored by `experiments/run_accuracy.py`, `run_accuracy_c3b.py` and `score_c1prime.py`. Meta-text in 14 C1 and 28 C1' translations was removed by `data_prep/extract_clean_questions.py` + `data_prep/apply_clean_questions.py`, and the cleaned conditions were re-scored by `experiments/rerun_cleaned_accuracy.py` (`*_clean.json`). The paper uses the clean versions.

## Mapping to the paper

| Paper section | Script(s) | Result file(s) |
|---|---|---|
| IV-A main accuracy, Table I, Fig. 1 | `paper/paper_numbers.py`, `paper/plot_figures.py` | `results/accuracy_results_200_clean.json`, `results/accuracy_c1prime_200_clean.json`, `results/accuracy_results_c3b_200.json` |
| IV-B which evidence the search returns | `paper/paper_numbers.py`; PubMedQA round trip: `experiments/run_pubmedqa_transfer.py`, `analysis/analyze_pubmedqa_transfer.py` | `results/results_pubmedqa_transfer_200.json` |
| III-C / IV-C matched-context factorial, Table II (GPT-4o-mini) | `experiments/matched_context_ablation.py` | `results/results_matched_context_200.json` |
| Table II (Claude Sonnet 4.6 replication) | `experiments/claude_2x2.py` | `results/results_claude_2x2_200.json` |
| IV-D translation-quality score (GEMBA-DA) | `experiments/exp3_qe.py --provider {claude,gpt}` | `results/exp3_qe_{claude,gpt}.json` |
| IV-D quality gate / agent check | `experiments/m4_agent_loop.py`, `analysis/analyze_review2.py` | `results/m4_relevance_gpt.json` |
| IV-E recovery without re-retrieval | `experiments/matched_context_ablation.py`, `paper/paper_numbers.py` | `results/results_matched_context_200.json` |
| IV-F medical retriever + corpus (MedCPT, MedRAG textbooks) | `experiments/m2_medcpt.py` | `results/results_m2_medcpt_200.json` |
| IV-F bypassing translation (multilingual MiniLM) and NLLB-200 | `experiments/m3_crosslingual_nllb.py`, `experiments/build_index_mminilm_v2.py` | `results/results_m3_200.json` |
| III-E error taxonomy (three annotators) | `annotation/` | `data/annotator{1,2,3}.xlsx`, `data/human_annotation_*` |
| VII memorization probes | `experiments/exp_memorization.py --provider {claude,gpt}` | `results/exp_memorization_{claude,gpt}.json` |

`experiments/review2_common.py` and `experiments/llm_client.py` are shared modules. Scripts not listed above (e.g. `run_c1b_control.py`, `run_c2a.py`, `run_selfcheck_*.py`, `run_norag_baseline.py`, `exp1_attribution.py`, `exp1b_grounding_judge.py`) are analyses from earlier versions of the paper; their outputs are kept in `results/` for completeness.

## Reproducing the numbers

Every number in the paper is computed by one script that reads the JSON files in `results/` and makes no API calls:

```bash
python paper/paper_numbers.py      # prints all statistics, writes results/paper_numbers_out.json
python paper/plot_figures.py       # writes paper/figures/fig1_accuracy.{pdf,png}
```

All McNemar tests are exact (two-sided binomial); intervals are percentile bootstrap over questions (10,000 resamples).

Re-running the experiments themselves requires API keys and the retrieval corpus (below).

## Setup

```bash
conda create -n rag-medical python=3.11
conda activate rag-medical
pip install openai anthropic python-dotenv sentence-transformers faiss-cpu transformers torch \
            datasets numpy scipy statsmodels pandas openpyxl matplotlib scikit-learn
```

Create a `.env` file in the repository root (it is git-ignored):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Data

**Questions.** The 200 questions (seed=42) from [CMB-Exam](https://github.com/FreedomIntelligence/CMB) (Apache-2.0) are included as `data/gold_set_200.json`, with translations in `data/gold_set_200_translated.json`. CMB itself is only needed to re-run the sampling step:

```bash
git clone https://github.com/FreedomIntelligence/CMB.git
# download the CMB data from https://huggingface.co/datasets/FreedomIntelligence/CMB into CMB/data/ so that
#   CMB/data/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json exists
python data_prep/resample_gold_set.py
```

**Retrieval corpus (not included, too large for GitHub).** The experiments expect three files in `data/`:

- `pubmed_abstracts.json` (~500 MB): a list of 852,618 passages from 273,458 PubMed abstracts in the PubMedQA source corpus, each `{"pubid": ..., "question": ..., "text": ...}`
- `pubmed_index.faiss`: FAISS `IndexFlatL2` over `all-MiniLM-L6-v2` embeddings of `text`, in the same order
- `pubmed_index_mminilm.faiss`: the same passages encoded with `paraphrase-multilingual-MiniLM-L12-v2` (built by `experiments/build_index_mminilm_v2.py`)

To rebuild them from PubMedQA (`pqa_labeled` + `pqa_unlabeled` + `pqa_artificial`, one passage per context section, duplicates removed):

```bash
python data_prep/build_pubmed_index.py            # corpus -> data/pubmed_abstracts_rebuilt.json
python data_prep/build_pubmed_index.py --index    # also encodes the corpus and builds data/pubmed_index_rebuilt.faiss
```

The original build script was not kept; `build_pubmed_index.py` was reconstructed from the description in Sec. III-A. It writes `*_rebuilt` files, never overwrites the originals, and compares its output with the existing files when they are present. Checked against the files used in the paper: the rebuilt corpus is identical (852,618 passages, same order), and the rebuilt index matches to within 1.6e-7 on 2,000 sampled vectors.

## Citation

```bibtex
@inproceedings{liang2026lost,
  title     = {Lost in the Question: Translation Errors and the Evidence They Retrieve in Cross-Lingual Medical RAG},
  author    = {Liang, Xuelin and Ju, Andrew},
  booktitle = {Proceedings of the First Workshop on Agentic Medical AI (AMAI), IEEE BIBM},
  year      = {2026}
}
```
