import json
import torch
from sentence_transformers import SentenceTransformer, util

# Load model
print("Loading sentence transformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load reference responses (human-written polite)
with open("test_f.json") as f:
    test_data = json.load(f)

references = [item["chosen"] for item in test_data]
ref_embeddings = model.encode(references, convert_to_tensor=True)

# Load all three response files
files = {
    "baseline": "baseline_responses.json",
    "prompted": "prompted_responses.json",
    "dpo":      "dpo_responses.json"
}

results = {}

for condition, filepath in files.items():
    with open(filepath) as f:
        data = json.load(f)

    condition_results = []

    for i, item in enumerate(data):
        response_embedding = model.encode(
            item["response"],
            convert_to_tensor=True
        )

        # Compare against the matching reference (same prompt id)
        ref_idx = next(
            (j for j, t in enumerate(test_data)
             if t["id"] == item["id"]), i % len(test_data)
        )

        similarity = util.cos_sim(
            response_embedding,
            ref_embeddings[ref_idx]
        ).item()

        row = {
            "id":                item["id"],
            "type":              item["type"],
            "cosine_similarity": round(similarity, 4)
        }

        # Keep phrasing info for prompted condition
        if "phrasing" in item:
            row["phrasing"] = item["phrasing"]

        condition_results.append(row)

    results[condition] = condition_results
    print(f"{condition} done")

# Save
with open("embedding_scores.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved to embedding_scores.json")