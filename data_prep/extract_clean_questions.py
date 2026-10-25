# extract_clean_questions.py
import json, os, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """The text below is the raw output of a Chinese-to-English translation request. \
It may contain the English translation surrounded by extra material (preambles such as \
"Certainly! Here is...", notes, separators, or the quoted Chinese source).

Return ONLY the English translation, copied verbatim. Do not rephrase, correct, complete, \
or translate anything. If the translation is wrapped in quotation marks, return it without them.

Text:
{text}"""

CJK = re.compile(r"[\u4e00-\u9fff]")

def norm(s):
    s = re.sub(r"[\"“”'‘’]", "", s)          # Ignore quote differences
    return re.sub(r"\s+", " ", s).strip()

def extract(text):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        temperature=0,
        max_tokens=500,
    )
    return r.choices[0].message.content.strip()

src = json.load(open("data/contaminated_questions.json", encoding="utf-8"))
out, n_manual = {}, 0
for fn, items in src.items():
    out[fn] = []
    for it in items:
        cleaned = extract(it["text"])
        is_sub = norm(cleaned) in norm(it["text"])
        has_cjk = bool(CJK.search(cleaned))
        status = "ok" if (is_sub and not has_cjk and cleaned) else "manual"
        n_manual += status == "manual"
        out[fn].append({"id": it["id"], "status": status,
                        "substring": is_sub, "cjk": has_cjk,
                        "original": it["text"], "cleaned": cleaned})
    ok = sum(x["status"] == "ok" for x in out[fn])
    print(f"{fn}: ok {ok}/{len(items)}")

json.dump(out, open("data/cleaned_questions_proposal.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\nsaved cleaned_questions_proposal.json  (manual: {n_manual})\n")

for fn, items in out.items():
    for x in items:
        if x["status"] == "manual":
            print(f"--- {fn} id={x['id']} substring={x['substring']} cjk={x['cjk']}")
            print("ORIGINAL:", x["original"][:500])
            print("CLEANED :", x["cleaned"][:500])