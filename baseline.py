import ollama
import json


#runs the 10 test prompts thru llama3.2 with nothing added
#The model will just respond however it naturally would
with open("test_f.json") as f:
    test_data = json.load(f)

results = []

for item in test_data:
    print(f"Running prompt {item['id']}...")

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "user", 
            "content": item["prompt"] +
            "Answer in 2 sentences maximum."
            }
        ],
        options={
            "num_predict": 100, # Fewer sentences
            "temperature": 0.0
        }
    )

    results.append({
        "id":       item["id"],
        "type":     item["type"],
        "prompt":   item["prompt"],
        "response": response["message"]["content"]
    })

with open("baseline_responses.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved to baseline_responses.json")