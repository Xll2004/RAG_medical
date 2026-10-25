import json
import os
import torch
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30)

print("Loading NLI model...")
nli_model_name = "cross-encoder/nli-deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
nli_model.eval()

def nli_contradiction_score(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = nli_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0][0].item()

def build_prompt(question, options, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"""You are a medical expert. Based on the following context, answer the question and select the correct option.

Context:
{context}

Question: {question}

Options:
{options_text}

Provide a brief explanation and indicate which option is correct."""

def get_sample(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=1.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Sample error: {e}")
        return ""

def selfcheck_bertscore(question, options, answer, retrieved_docs, n_samples=3):
    prompt = build_prompt(question, options, retrieved_docs)
    samples = [get_sample(prompt) for _ in range(n_samples)]
    samples = [s for s in samples if s]
    if not samples:
        return 0.5
    references = [answer] * len(samples)
    P, R, F1 = bert_score(samples, references, lang="en", verbose=False)
    return 1 - F1.mean().item()

def selfcheck_nli(question, options, answer, retrieved_docs, n_samples=3):
    prompt = build_prompt(question, options, retrieved_docs)
    samples = [get_sample(prompt) for _ in range(n_samples)]
    samples = [s for s in samples if s]
    if not samples:
        return 0.5
    scores = [nli_contradiction_score(answer, sample) for sample in samples]
    return sum(scores) / len(scores)

# Load results
with open("results_condition3b_200.json") as f:
    c3b = json.load(f)

n = len(c3b)
print(f"Using {n} questions")
print("Running SelfCheckGPT (BERTScore + NLI) on C3b (200 questions)...")

bert_scores = []
nli_scores = []

for i in range(n):
    print(f"Q{i+1}/{n}...", end=" ", flush=True)

    bert_s = selfcheck_bertscore(
        c3b[i]["question_corrected"],
        c3b[i]["options_corrected"],
        c3b[i]["answer"],
        c3b[i]["retrieved_docs"]
    )
    nli_s = selfcheck_nli(
        c3b[i]["question_corrected"],
        c3b[i]["options_corrected"],
        c3b[i]["answer"],
        c3b[i]["retrieved_docs"]
    )

    bert_scores.append({"id": c3b[i]["id"], "score_c3b": bert_s})
    nli_scores.append({"id": c3b[i]["id"], "score_c3b": nli_s})

    print(f"BERTScore:{bert_s:.2f} NLI:{nli_s:.2f}")

    with open("selfcheck_scores_c3b_bertscore_200.json", "w") as f:
        json.dump(bert_scores, f, indent=2)
    with open("selfcheck_scores_c3b_nli_200.json", "w") as f:
        json.dump(nli_scores, f, indent=2)

# Load existing results for comparison
with open("selfcheck_scores_bertscore_200.json") as f:
    bert_data = json.load(f)
with open("selfcheck_scores_nli_200.json") as f:
    nli_data = json.load(f)

print("\n=== Final Results (200 questions) ===")
print(f"{'Condition':<20} {'BERTScore':>10} {'NLI':>10}")
print("-" * 42)
print(f"{'C2 MT':<20} {np.mean([s['score_c2'] for s in bert_data]):>10.3f} {np.mean([s['score_c2'] for s in nli_data]):>10.3f}")
print(f"{'C3 General':<20} {np.mean([s['score_c3'] for s in bert_data]):>10.3f} {np.mean([s['score_c3'] for s in nli_data]):>10.3f}")
print(f"{'C3b Terminology':<20} {np.mean([s['score_c3b'] for s in bert_scores]):>10.3f} {np.mean([s['score_c3b'] for s in nli_scores]):>10.3f}")