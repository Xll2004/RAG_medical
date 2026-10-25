from transformers import MarianMTModel, MarianTokenizer
import json

# Load translation model
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    if not text or not text.strip():
        return ""
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Load gold set
with open("data/gold_set_200.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Translate questions and options
print("Translating 200 questions and options...")
for i, item in enumerate(data):
    item["question_mt"] = translate(item["question"])
    item["option_mt"] = {}
    for key, value in item["option"].items():
        item["option_mt"][key] = translate(value) if value.strip() else ""
    if (i + 1) % 20 == 0:
        print(f"{i + 1}/200 done")

# Save
with open("data/gold_set_200_translated.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Done. Saved to gold_set_200_translated.json")