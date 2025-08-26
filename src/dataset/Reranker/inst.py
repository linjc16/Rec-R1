import re
import os
from datasets import Dataset, load_dataset, concatenate_datasets
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from collections import defaultdict, Counter
import random
import pdb
import csv

PROMPT = """You are an expert in product reranking. 
Given a customer query and a list of candidate products, 
your task is to rerank the products so that the most relevant items to the query are placed at the top. 
Consider semantic meaning, product attributes, and customer intent.

Below is the query:
```{user_query}```

Below are the candidate products (item_id and metadata):
```{candidates}```
------
"""


def make_prefix(dp, template_type='qwen'):

    input_str = PROMPT.format(user_query=dp['query'], candidates=dp['candidates'])
    
    if template_type == 'qwen':
        input_str = (
            "<|im_start|>system\nYou are a helpful AI assistant. "
            "You first think about the reasoning process in the mind and then provide the user with the answer."
            "<|im_end|>\n<|im_start|>user\n" 
            + input_str +
            """\nShow your work in <think> </think> tags. 
Your final response must be in JSON format within <answer> </answer> tags. 
The output JSON should contain the reranked list of item_ids in order of relevance. You must include **all the input candidates** exactly once (no missing items, no duplicates).
For example,
<answer>
{
    "reranked_items": ["B001", "B002", "B003", ...]
}
</answer><|im_end|>
<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
        )
    elif template_type == 'llama3':
        input_str = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant. "
            "You first think about the reasoning process in the mind and then provide the user with the answer."
            "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n" 
            + input_str +
            """\nPlease show your entire reasoning process in **a single** <think> </think> block. 
Your final response must be in JSON format within <answer> </answer> tags. 
The output JSON should contain the reranked list of item_ids in order of relevance. You must include **all the input candidates** exactly once (no missing items, no duplicates).
For example,
<think>
[entire reasoning process here]
</think>
<answer>
{
    "reranked_items": ["B001", "B002", "B003"]
}
</answer><|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\nLet me solve this step by step.\n<think>"""
        )
    return input_str


def load_rec_dataset(data_dir, domain_name_list):
    with open(os.path.join(data_dir, 'train', f'esci.jsonl'), 'r') as f:
        train_data = [json.loads(line) for line in f]
    
    with open(os.path.join(data_dir, 'val', f'esci.jsonl'), 'r') as f:
        val_data = [json.loads(line) for line in f]

    test_data_dict = {}
    for domain_name in domain_name_list:
        with open(os.path.join(data_dir, 'test', f'{domain_name}.jsonl'), 'r') as f:
            test_data = [json.loads(line) for line in f]
        
        test_data_dict[domain_name] = test_data

    item2metadata = {}
    with open("data/esci/raw/sampled_item_metadata_esci.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = obj["item_id"]
            metadata = obj["metadata"]
            item2metadata[item_id] = metadata
    
    # def function that convert input item list to candidates string with item_id and metadata
    def convert_to_candidates(item_list):
        candidates = []
        for item_id in item_list:
            metadata = item2metadata.get(item_id, "No metadata available")
            candidates.append(f"Item ID: {item_id}\nMetadata: {metadata}")
        return "\n\n".join(candidates)

    # process the data with pandas in parallel
    for dp in tqdm(train_data):
        dp['candidates'] = convert_to_candidates(dp['input'])
    for dp in tqdm(val_data):
        dp['candidates'] = convert_to_candidates(dp['input'])
    for domain_name in domain_name_list:
        for dp in tqdm(test_data_dict[domain_name]):
            dp['candidates'] = convert_to_candidates(dp['input'])
    

    return train_data, val_data, test_data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/reranker/raw')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, choices=['qwen', 'llama3'], default='qwen')
    parser.add_argument('--save_dir', type=str, default='data/reranker/inst/')
    
    args = parser.parse_args()
    
    domain_name_list = ['Video_Games', 'Baby_Products', 'Office_Products', 'Sports_and_Outdoors']
    
    data_source = f'reranker'
    data_source_dict = {ele: f'reranker_{ele}' for ele in domain_name_list}

    file_dir = args.local_dir
    save_dir = os.path.join(args.save_dir, args.template_type)
    os.makedirs(save_dir, exist_ok=True)
    
    train_data, val_data, test_data_dict = load_rec_dataset(file_dir, domain_name_list)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    test_dataset_dict = {}
    for domain_name in domain_name_list:
        test_data = test_data_dict[domain_name]
        test_dataset_dict[domain_name] = Dataset.from_list(test_data)
    
    
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = make_prefix(example, args.template_type)
            solution = {
                "target": example['target'],
            }
            data = {
                "data_source": data_source + f'_{split}',
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "reranking",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('val', data_source), with_indices=True)

    for domain_name in domain_name_list:
        test_dataset = test_dataset_dict[domain_name]
        test_dataset_dict[domain_name] = test_dataset.map(function=make_map_fn('test', data_source_dict[domain_name]), with_indices=True)

    # concat val_dataset, all test_dataset
    test_dataset = concatenate_datasets(list(test_dataset_dict.values()))
    # # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)

    # concatenate val_1 and val_2
    val_dataset = concatenate_datasets([val_dataset, test_dataset])
    
    threshold = 2048
    
    def truncate(train_dataset, threshold):
        count = 0
        # for those that are exceeding the threshold, we can delete the text between "\nTitle:" to "\nInclusion criteria:"
        for i, d in enumerate(train_dataset):
            if len(d['prompt'][0]['content'].split()) > threshold:
                text = d['prompt'][0]['content']
                count += 1
                words = text.split()
                truncate_length = max(threshold - 200, 0)  # Ensure we don't end up with a negative index
                text = ' '.join(words[-truncate_length:])
                train_dataset[i]['prompt'][0]['content'] = text

        print(f"Truncated {count} examples")
    
        return train_dataset
    
    train_dataset = truncate(train_dataset, threshold=threshold)

    hdfs_dir = os.path.join(args.hdfs_dir, args.template_type) if args.hdfs_dir is not None else None
    
    pdb.set_trace()

    train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(save_dir, 'val.parquet'))
    test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=save_dir, dst=hdfs_dir) 
