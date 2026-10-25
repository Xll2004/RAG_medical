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

sample = annotations

def classify_error_claude(question_zh, question_mt, option_zh, option_mt):
    prompt = f"""Original Chinese question: {question_zh}
Machine translated English: {question_mt}
Original Chinese options: {json.dumps(option_zh, ensure_ascii=False)}
Machine translated options: {json.dumps(option_mt, ensure_ascii=False)}

Classify translation errors using these codes:
1 = Terminology error: medical terms mistranslated
2 = Syntactic distortion: sentence structure broken
3 = Semantic drift: overall meaning shifted
0 = No errors

Reply with ONLY valid JSON, no explanation outside JSON:
{{"error_types": ["1", "3"], "explanation": "brief reason"}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system="You are a translation error classifier. You must respond with ONLY a valid JSON object and nothing else. No markdown, no preamble.",
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {"error_types": ["s"], "explanation": "parse error"}

results = []
agree = 0

for i, item in enumerate(sample):
    print(f"Q{i+1}/200...", end=" ", flush=True)
    q = questions[item["id"]]

    claude_result = classify_error_claude(
        q["question"], q["question_mt"],
        q["option"], q["option_mt"]
    )

    gpt_types = set(item["error_types"])
    cl_types = set(claude_result["error_types"])
    matched = gpt_types == cl_types

    if matched:
        agree += 1

    results.append({
        "id": item["id"],
        "gpt_annotation": item["error_types"],
        "claude_annotation": claude_result["error_types"],
        "agreement": matched
    })

    print(f"GPT: {item['error_types']} | Claude: {claude_result['error_types']} | {'✓' if matched else '✗'}")

agreement_rate = agree / len(sample) * 100
print(f"\n=== Agreement: {agree}/{len(sample)} = {agreement_rate:.1f}% ===")

with open("spotcheck_agreement_200.json", "w", encoding="utf-8") as f:
    json.dump({"agreement_rate": agreement_rate, "results": results}, f, ensure_ascii=False, indent=2)
print("Saved spotcheck_agreement_200.json")
