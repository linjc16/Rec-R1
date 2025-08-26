import json
import os

def convert_jsonl_for_pyserini(input_file, output_file):
    """Convert JSONL data to Pyserini-compatible format with a structured 'contents' field"""
    docs = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())

            # Create JSON document with a clear structure
            doc = {
                "id": data["_id"],  
                "contents": data['text'].strip(),
            }
            
            docs.append(json.dumps(doc))

    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n")
    
    print(f"✅ Converted JSONL saved to {output_file}")


ori_data_dir = "data/dl19/raw/corpus/human.jsonl"
output_file = "database/dl1920/jsonl_docs/dl1920_metadata.jsonl"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Example Usage
convert_jsonl_for_pyserini(ori_data_dir, output_file)
