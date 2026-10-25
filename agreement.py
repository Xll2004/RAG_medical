import json
import re
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

with open("error_annotations_200.json", "r", encoding="utf-8") as f:
    annotations = json.load(f)

with open("gold_set_200_translated.json", "r", encoding="utf-8") as f:
    questions = {item["id"]: item for item in json.load(f)}

def classify_error_claude(question_zh, question_mt, option_zh, option_mt, debug=False):
    prompt = f"""Analyze the machine translation quality of this Chinese medical question.

Original Chinese question: {question_zh}
Machine translated English: {question_mt}

Original Chinese options: {json.dumps(option_zh, ensure_ascii=False)}
Machine translated options: {json.dumps(option_mt, ensure_ascii=False)}

Error categories:
1 = Terminology error: medical terms are mistranslated
2 = Syntactic distortion: sentence structure is broken or meaning is lost
3 = Semantic drift: the overall meaning has shifted significantly

You MUST respond with ONLY a JSON object, no other text before or after.
Example: {{"error_types": ["1", "3"], "explanation": "brief reason"}}
If no errors: {{"error_types": ["0"], "explanation": "translation is accurate"}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system="You are a translation error classifier. Your entire response must be a single valid JSON object. Do not include markdown, preamble, or any text outside the JSON.",
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text).strip()
    if debug:
        print(f"\nDEBUG raw response: {text}\n")
    try:
        return json.loads(text)
    except:
        pass
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    try:
        types_match = re.search(r'"error_types"\s*:\s*(\[[^\]]+\])', text)
        if types_match:
            types = json.loads(types_match.group(1))
            return {"error_types": types, "explanation": "extracted"}
    except:
        pass
    print(f"\nPARSE FAIL raw: {text[:200]}\n")
    return {"error_types": ["s"], "explanation": "parse error"}

results = []
jaccard_scores = []
n = len(annotations)

for i, item in enumerate(annotations):
    print(f"Q{i+1}/{n}...", end=" ", flush=True)
    q = questions[item["id"]]

    claude_result = classify_error_claude(
        q["question"],
        q["question_mt"],
        q["option"],
        q["option_mt"],
        debug=(i == 0)
    )

    gpt = set(item["error_types"]) - {"s"}
    cl = set(claude_result["error_types"]) - {"s"}

    if not gpt and not cl:
        jaccard = 1.0
    elif not gpt or not cl:
        jaccard = 0.0
    else:
        jaccard = len(gpt & cl) / len(gpt | cl)

    jaccard_scores.append(jaccard)
    results.append({
        "id": item["id"],
        "gpt_annotation": item["error_types"],
        "claude_annotation": claude_result["error_types"],
        "jaccard": jaccard
    })

    print(f"GPT: {item['error_types']} | Claude: {claude_result['error_types']} | J={jaccard:.2f}")

avg_jaccard = sum(jaccard_scores) / n
print(f"\n=== Average Jaccard (n=200): {avg_jaccard:.3f} ===")

with open("spotcheck_agreement_200.json", "w", encoding="utf-8") as f:
    json.dump({"avg_jaccard": avg_jaccard, "n": n, "results": results}, f, ensure_ascii=False, indent=2)

print("Saved spotcheck_agreement_200.json")