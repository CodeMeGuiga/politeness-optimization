import json
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Intel/polite-guard",
    top_k=None
)

# Load all three response files
files = {
    "baseline":  "baseline_responses.json",
    "prompted":  "prompted_responses.json",
    "dpo":       "dpo_responses.json"
}

results = {}

for condition, filepath in files.items():
    with open(filepath) as f:
        data = json.load(f)
    
    condition_results = []
    
    for item in data:
        scores = classifier(item["response"][:512])[0]
        top = max(scores, key=lambda x: x["score"])
        
        condition_results.append({
            "id":        item["id"],
            "type":      item["type"],
            "label":     top["label"],
            "score":     round(top["score"], 4),
            "all":       {s["label"]: round(s["score"], 4) 
                         for s in scores},
            "phrasing":  item.get("phrasing", None)
        })
    
    results[condition] = condition_results
    print(f"{condition} done")

with open("bert_scores.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved to bert_scores.json")