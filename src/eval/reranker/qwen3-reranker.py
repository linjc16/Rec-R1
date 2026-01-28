import os
import re
import gc
import math
import json
import argparse
from typing import List, Tuple, Dict

from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.distributed.parallel_state import destroy_model_parallel

# -----------------------------
# Dataset loader
# -----------------------------
def load_rec_dataset(data_dir, domain_name_list):
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
    
    return test_data_dict, item2metadata


# -----------------------------
# Reranker helpers (vLLM)
# -----------------------------
def format_instruction(instruction, query, doc):
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]
    return text

def process_inputs(tokenizer, pairs, instruction, max_length, suffix_tokens):
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    token_seqs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    token_seqs = [seq[:max_length] + suffix_tokens for seq in token_seqs]
    return [TokensPrompt(prompt_token_ids=seq) for seq in token_seqs]

def compute_logits(model, messages, sampling_params, true_token, false_token):
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    return scores


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/reranker/raw")
    parser.add_argument("--save_dir", type=str, default="results/reranker/qwen3-8b")
    parser.add_argument("--cache_dir", type=str, default="/srv/local/data/linjc/hub")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_mem_util", type=float, default=0.8)
    parser.add_argument("--batch_queries", type=int, default=1)
    parser.add_argument("--domains", type=str, nargs="*", default=["Video_Games", "Baby_Products", "Office_Products", "Sports_and_Outdoors"])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Load test data (domain → list of datapoints)
    test_dict, item2metadata = load_rec_dataset(args.data_dir, args.domains)
    
    # 2) Load reranker
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-8B', cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = LLM(model='Qwen/Qwen3-Reranker-8B',
                tensor_parallel_size=1,
                max_model_len=args.max_model_len,
                enable_prefix_caching=True,
                gpu_memory_utilization=args.gpu_mem_util)

    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    instruction = "Given a web search query, retrieve relevant passages that answer the query"

    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    # 3) Loop over each domain separately
    for domain, dataset in test_dict.items():
        out_path = os.path.join(args.save_dir, f"qwen3_rerank_test_{domain}.jsonl")
        fw = open(out_path, "w", encoding="utf-8")

        for start in tqdm(range(0, len(dataset), args.batch_queries), desc=f"Reranking {domain}"):
            batch = dataset[start:start + args.batch_queries]

            # flatten pairs
            flat_pairs = []
            segs = []
            for dp in batch:
                q = dp["query"]
                item_ids = dp["input"][:16]
                docs = [item2metadata.get(i, "No metadata available") for i in item_ids]
                pairs = [(q, d) for d in docs]
                segs.append((len(flat_pairs), len(pairs), item_ids, dp["target"], q))
                flat_pairs.extend(pairs)

            messages = process_inputs(tokenizer, flat_pairs, instruction, args.max_model_len - len(suffix_tokens), suffix_tokens)
            scores = compute_logits(model, messages, sampling_params, true_token, false_token)

            for (offset, length, item_ids, target, q) in segs:
                my_scores = scores[offset:offset+length]
                ranked = [x for _, x in sorted(zip(my_scores, item_ids), key=lambda z: z[0], reverse=True)]
                rec = {
                    "query": q,
                    "ranked": ranked,
                    "target": target
                }
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

        fw.close()
        print(f"[OK] Saved domain {domain} results to {out_path}")

    destroy_model_parallel()
    gc.collect()


if __name__ == "__main__":
    main()
