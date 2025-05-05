import json

def save_to_json(data, filename):
    if isinstance(data, str):
        data = data.strip()
        if data.startswith("```json"):
            data = data[7:].strip()
        if data.endswith("```"):
            data = data[:-3].strip()

        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            print(f"❌ Cannot decode JSON string. Error: {e}")
            return
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ Data saved to {filename}")