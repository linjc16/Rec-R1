#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Optional, List, Dict, Any

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate query rewrites with doc2query for amazon_c4")
    parser.add_argument('--local_dir', default='data/amazon_c4/subset',
                        help='Directory containing domain subfolders (each has test.json, train.json)')
    parser.add_argument('--domain_name_list', nargs='*',
                        default=['Video_Games', 'Baby', 'Office', 'Sports'],
                        help='Domain subfolder names to process (each must have test.json)')
    parser.add_argument('--save_dir', default='results/amazon_c4/doc2query',
                        help='Output directory')
    parser.add_argument('--num_rewrites', type=int, default=5,
                        help='Number of rewrites per query')
    parser.add_argument('--model_name', default='doc2query/msmarco-t5-base-v1',
                        help='HuggingFace model name')
    parser.add_argument('--cache_dir', default=None,
                        help='HF cache dir (e.g., /srv/local/data/linjc/hub)')
    parser.add_argument('--max_input_len', type=int, default=320,
                        help='Max input length')
    parser.add_argument('--max_gen_len', type=int, default=64,
                        help='Max generated length')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p for sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_rewrites(text: str,
                      tokenizer: T5Tokenizer,
                      model: T5ForConditionalGeneration,
                      device: str,
                      num_rewrites: int,
                      max_input_len: int,
                      max_gen_len: int,
                      top_p: float,
                      temperature: float) -> List[str]:
    input_ids = tokenizer.encode(
        text, max_length=max_input_len, truncation=True, return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_gen_len,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_rewrites
        )

    seen, uniq = set(), []
    for i in range(outputs.size(0)):
        q = tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
        q = q.replace('\t', ' ').replace('\n', ' ').strip()
        if q and q.lower() not in seen:
            uniq.append(q)
            seen.add(q.lower())
    return uniq


def read_records_from_json(path: str) -> List[Dict[str, Any]]:
    """Expect a JSON list, each item having at least 'query' and 'item_id'."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [obj for obj in data if isinstance(obj, dict) and "query" in obj]
        else:
            print(f"[WARN] {path} is not a list; skipping")
            return []
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return []


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer, model, device = load_model_and_tokenizer(args.model_name, args.cache_dir)

    overall = {}

    for domain in args.domain_name_list:
        json_path = os.path.join(args.local_dir, domain, "test.json")
        if not os.path.isfile(json_path):
            print(f"[WARN] File not found: {json_path}; skipping")
            continue

        print(f"[INFO] Processing {json_path}")
        records = read_records_from_json(json_path)
        if not records:
            print(f"[WARN] No records in {json_path}; skipping")
            continue

        domain_result = {}
        for obj in tqdm(records, desc=f"Generating rewrites for {domain}"):
            query = str(obj.get("query", "")).strip()
            item_ids = obj.get("item_id", [])
            if not query:
                continue

            rewrites = generate_rewrites(
                text=query,
                tokenizer=tokenizer,
                model=model,
                device=device,
                num_rewrites=args.num_rewrites,
                max_input_len=args.max_input_len,
                max_gen_len=args.max_gen_len,
                top_p=args.top_p,
                temperature=args.temperature
            )

            domain_result[query] = {
                "rewrites": rewrites,
                "item_id": item_ids if isinstance(item_ids, list) else [item_ids]
            }
            overall.setdefault(query, domain_result[query])

        save_path = os.path.join(args.save_dir, domain, "rewrites.json")
        save_json(domain_result, save_path)
        print(f"[INFO] Saved: {save_path}")

    all_save_path = os.path.join(args.save_dir, "_all", "rewrites.json")
    save_json(overall, all_save_path)
    print(f"[INFO] Saved summary: {all_save_path}")


if __name__ == "__main__":
    main()
