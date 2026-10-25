import json
import os
import re
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_error(question_zh, question_mt, option_zh, option_mt):
    prompt = f"""You are a translation error analyst. Analyze the machine translation quality of this Chinese medical question.

Original Chinese question: {question_zh}
Machine translated English: {question_mt}

Original Chinese options: {json.dumps(option_zh, ensure_ascii=False)}
Machine translated options: {json.dumps(option_mt, ensure_ascii=False)}

Classify the translation errors into these categories:
1 = Terminology error: medical terms are mistranslated
2 = Syntactic distortion: sentence structure is broken or meaning is lost
3 = Semantic drift: the overall meaning has shifted significantly

Reply with ONLY a JSON object like this:
{{"error_types": ["1", "3"], "explanation": "brief reason"}}

If no errors, use: {{"error_types": ["0"], "explanation": "translation is accurate"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )
    text = response.choices[0].message.content.strip()
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {"error_types": ["s"], "explanation": "parse error"}

# Load data
with open("gold_set_200_translated.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Auto-classifying translation errors for 200 questions...")
annotations = []

for i, item in enumerate(data):
    print(f"Q{i+1}/200...", end=" ", flush=True)

    result = classify_error(
        item["question"],
        item["question_mt"],
        item["option"],
        item["option_mt"]
    )

    annotations.append({
        "id": item["id"],
        "question_zh": item["question"],
        "question_mt": item["question_mt"],
        "error_types": result["error_types"],
        "explanation": result["explanation"]
    })

    print(f"Types: {result['error_types']}")

    with open("error_annotations_200.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

# Summary
error_questions = defaultdict(list)
for idx, a in enumerate(annotations):
    for t in a["error_types"]:
        error_questions[t].append(idx + 1)

print("\n=== Error Type Distribution (200 questions) ===")
for t, label in [("1", "T1 Terminology error"), ("2", "T2 Syntactic distortion"),
                 ("3", "T3 Semantic drift"), ("0", "No error")]:
    q_nums = error_questions[t]
    print(f"\n{label}: {len(q_nums)}/200")