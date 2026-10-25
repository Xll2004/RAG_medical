# apply_clean_questions.py
import ast, json
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

# Rule: only delete meta-text (preambles, notes, "The translation of..." wrappers); never change the translation itself (including inline Chinese notes)
KEEP_ORIGINAL = {
    "results/results_condition1_200.json": {10963, 773, 4351, 1939, 9350, 2934},
    "results/results_c1prime_200.json":    {9350, 1445, 9462},
}
# Manual entries: deletions only on the original text, no rewriting
MANUAL = {
    "results/results_condition1_200.json": {
        7913: lambda s: s.split("\n\n(Note:")[0],
        9462: lambda s: s.split(" This highlights")[0],
    },
    "results/results_c1prime_200.json": {
        117:   lambda s: s.split("---\n\n", 1)[1],
        11106: lambda s: s.split("---\n\n")[1]
                          .replace(" (syndrome differentiation ends here).", ""),
    },
}

FILES = {
    "results/results_condition1_200.json": ("question_en_baseline", "experiments/run_condition1.py"),
    "results/results_c1prime_200.json":    ("question_en_gpt41",    "experiments/run_c1prime.py"),
}

print("Loading index / abstracts / encoder ...")
index = faiss.read_index("data/pubmed_index.faiss")
abstracts = json.load(open("data/pubmed_abstracts.json", encoding="utf-8"))
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def load_retrieve(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "retrieve")
    ns = {"index": index, "abstracts": abstracts, "encoder": encoder, "np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns["retrieve"]

proposal = json.load(open("data/cleaned_questions_proposal.json", encoding="utf-8"))
log = {}

for fn, (key, script) in FILES.items():
    retrieve = load_retrieve(script)
    data = json.load(open(fn, encoding="utf-8"))
    by_id = {x["id"]: x for x in data}

    fixes = {}
    for p in proposal[fn]:
        if p["id"] in KEEP_ORIGINAL[fn]:
            continue
        if p["id"] in MANUAL[fn]:
            fixes[p["id"]] = MANUAL[fn][p["id"]](p["original"]).strip()
        elif p["status"] == "ok":
            fixes[p["id"]] = p["cleaned"]
        else:
            raise ValueError(f"Unhandled manual entry: {fn} id={p['id']}")

    # Sanity: re-retrieve unchanged questions; results must match the saved ones exactly
    unchanged = [x for x in data if x["id"] not in fixes][:5]
    for x in unchanged:
        assert retrieve(x[key]) == x["retrieved_docs"], f"{fn} id={x['id']} retrieval not reproducible, aborting"
    print(f"\n{fn}: retrieval reproducibility check passed (5/5)")

    changed = []
    for qid, new_q in fixes.items():
        item = by_id[qid]
        new_docs = retrieve(new_q)
        overlap = len(set(new_docs) & set(item["retrieved_docs"]))
        changed.append({"id": qid, "old_q": item[key], "new_q": new_q, "doc_overlap": overlap})
        item[key + "_raw"] = item[key]
        item["retrieved_docs_raw"] = item["retrieved_docs"]
        item[key] = new_q
        item["retrieved_docs"] = new_docs
        item["cleaned"] = True

    out_fn = fn.replace(".json", "_clean.json")
    json.dump(data, open(out_fn, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log[fn] = changed

    print(f"{fn}: cleaned {len(changed)} questions -> {out_fn}")
    print(f"  docs unchanged (3/3): {sum(c['doc_overlap'] == 3 for c in changed)}"
          f" | partial: {sum(0 < c['doc_overlap'] < 3 for c in changed)}"
          f" | fully changed: {sum(c['doc_overlap'] == 0 for c in changed)}")
    for c in changed:
        print(f"  id={c['id']} overlap={c['doc_overlap']}/3")
        print(f"    OLD: {c['old_q'][:150]!r}")
        print(f"    NEW: {c['new_q'][:150]!r}")

json.dump(log, open("data/cleaning_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nsaved cleaning_log.json")