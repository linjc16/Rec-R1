import os
import json
import argparse
from datasets import Dataset
import pdb

# -------------------------------
# make_prefix function (same as before)
# -------------------------------
PROMPT = """You are an expert in query generation. Given a query, your task is to create query terms to retrieve retrieve the most relevant documents.
Below is the query:
```{user_query}```"""

def make_prefix(dp, template_type='qwen'):
    input_str = PROMPT.format(user_query=dp['query'])
    
    if template_type == 'qwen':
        input_str = """<|im_start|>system\nYou are a helpful AI assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n""" + input_str
        input_str += """\nShow your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. The generated query should use Boolean operators (AND, OR) to structure your query logically. For example,
<answer>
{
    "query": xxx
}
</answer><|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    elif template_type == 'llama3':
        input_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n""" + input_str
        input_str += """\nPlease show your entire reasoning process in **a single** <think> </think> block (do not open or close the tag more than once). Your final response must be in JSON format within <answer> </answer> tags. The generated query should use Boolean operators (AND, OR) to structure your query logically. For example,
<think>
[entire reasoning process here]
</think>
<answer>
{
    "query": xxx
}
</answer><|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\nLet me solve this step by step.\n<think>"""
    return input_str


# -------------------------------
# parse_qrel (your preferred version)
# -------------------------------
def parse_qrel(qrel_lines):
    """
    Parse a TSV file and return a dictionary where:
    - Keys are query IDs.
    - Values are dictionaries containing:
        - 'targets': list of relevant corpus IDs (sorted by score desc)
        - 'scores': list of integer scores (sorted desc)
    """
    query_dict = {}

    # Skip the header
    for line in qrel_lines[1:]:
        query_id, corpus_id, score = line.strip().split("\t")

        if score == '0':
            continue

        if query_id not in query_dict:
            query_dict[query_id] = {"targets": [], "scores": []}

        query_dict[query_id]["targets"].append(corpus_id)
        query_dict[query_id]["scores"].append(int(score))

    # Sort targets and scores by score (descending)
    for qid, data in query_dict.items():
        pairs = list(zip(data["targets"], data["scores"]))
        pairs.sort(key=lambda x: x[1], reverse=True)  # sort by score
        query_dict[qid]["targets"], query_dict[qid]["scores"] = zip(*pairs)
        query_dict[qid]["targets"] = list(query_dict[qid]["targets"])
        query_dict[qid]["scores"] = list(query_dict[qid]["scores"])

    return query_dict



# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", type=str, default=f"data/dl20/raw/queries.jsonl")
    parser.add_argument("--qrel_path", type=str, default="data/dl20/raw/qrels/test.tsv")
    parser.add_argument("--template_type", type=str, choices=["qwen", "llama3"], default="qwen")
    parser.add_argument("--save_dir", type=str, default="data/dl20/inst/sparse")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load queries.jsonl
    queries = {}
    with open(args.query_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"]

    # Load qrels from test.tsv
    with open(args.qrel_path, "r", encoding="utf-8") as f:
        qrels = parse_qrel(f.readlines())

    # Build test set
    test_data = []
    for qid, value in qrels.items():
        if qid not in queries:
            continue
        query_text = queries[qid]
        test_data.append({
            "qid": qid,
            "query": query_text,
            "item_id": value["targets"],
            "scores": value["scores"]
        })

    # Convert to HuggingFace Dataset
    test_dataset = Dataset.from_list(test_data)

    # Map to prefix format
    def process_fn(example, idx):
        question = make_prefix(example, args.template_type)
        solution = {
            "target": example["item_id"],
            "scores": example["scores"]
        }
        return {
            "data_source": f"dl19_test",
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "trec_dl",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                "split": "test",
                "index": idx,
                "qid": example["qid"]
            }
        }

    test_dataset = test_dataset.map(function=process_fn, with_indices=True)
    
    # Save to parquet
    save_path = os.path.join(args.save_dir, "test.parquet")
    test_dataset.to_parquet(save_path)
    print(f"Saved test set to {save_path}, total {len(test_dataset)} examples")
