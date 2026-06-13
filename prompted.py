import ollama
import json

with open("test_f.json") as f:
    test_data = json.load(f)


#------- Take this off later --------
# runs the exact same 10 prompts through the exact same model, 
# but this time adds a system prompt before each conversation telling it to be polite. 
# It does this three times with three different phrasings (A, B, C) to test whether the model's 
# behaviour changes depending on how the instruction is worded. If the outputs vary a lot across A, B, and C
# , that means prompting is fragile. If they are consistent, prompting is robust.
prompts = {
    "A": "Respond politely.",
    "B": "Respond in a polite and considerate manner, \
acknowledging the user's situation before giving advice.",
    "C": "Be warm, thoughtful, and sensitive in your response. \
Acknowledge what the person is going through, soften any \
suggestions, and match your tone to the context."
}

results = []

for phrasing, system_prompt in prompts.items():
    print(f"\nRunning phrasing {phrasing}...")
    for item in test_data:
        print(f"  Prompt {item['id']}...")

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", 
                "content": system_prompt},
                {"role": "user", 
                "content": item["prompt"] + "Answer in 2 sentences maximum."
                }
            ],
            options={
                "num_predict": 100,
                "temperature": 0.0
            }
        )

        results.append({
            "id":            item["id"],
            "type":          item["type"],
            "prompt":        item["prompt"],
            "phrasing":      phrasing,
            "system_prompt": system_prompt,
            "response":      response["message"]["content"]
        })

with open("prompted_responses.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved to prompted_responses.json")