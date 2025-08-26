import argparse
import pandas as pd
import json
import os
import re
import pdb
import ast
import sys
sys.path.append('./')

from src.eval_search.utils import ndcg_at_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', type=str, choices=['Video_Games', 'Baby_Products', 'Office_Products', 'Sports_and_Outdoors', 'esci'], default='esci')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='train')
    args = parser.parse_args()
    

    # load data/esci/inst/sparse/subset/{args.split}.parquet
    df = pd.read_parquet(f'data/esci/inst/sparse/subset/{args.split}.parquet')
    df = df[df['data_source'].str.contains(args.domain_name, case=False, na=False)]

    if args.split in ['train', 'val']:
        # load results/esci/metric_res/{args.split}/query_metric_results-rec-r1.json
        with open(f'results/esci/metric_res/{args.split}/query_metric_results-rec-r1.json', 'r') as f:
            results = json.load(f)
    elif args.split == 'test':
        # load results/esci/metric_res/query_metric_results-rec-r1-{args.domain_name}.json
        with open(f'results/esci/metric_res/query_metric_results-rec-r1-{args.domain_name}.json', 'r') as f:
            results = json.load(f)
    else:
        raise ValueError(f"Invalid split: {args.split}")

    data_points = []
    ndcg10_list = []

    for idx, (key, value) in enumerate(results.items()):
        # Skip if ndcg@20 == 0
        if value.get("ndcg@20", 0.0) == 0.0:
            continue
        
        # Get query from df (aligned by id/index)
        sample_id = int(value["id"])
        query = df.loc[sample_id]["query"]

        # Parse retrieved and target lists (they are string repr of lists in results)
        retrieved = ast.literal_eval(value["retrieved"]) if isinstance(value["retrieved"], str) else value["retrieved"]
        target = ast.literal_eval(value["target"]) if isinstance(value["target"], str) else value["target"]

        df_target = df.loc[sample_id]["item_id"]
        if not isinstance(df_target, list):
            df_target = list(df_target)

        if not isinstance(target, list):
            target = list(target)

        assert df_target == target, f"Mismatch in target at index {sample_id}"

        # Take top 16 retrieved items
        retrieved = retrieved[:16]

        ndcg10 = ndcg_at_k(retrieved, target, 10, [1] * len(target))
        ndcg10_list.append(ndcg10)

        data_points.append({
            "query": query,
            "input": retrieved,
            "target": target
        })
    
    print(f"Constructed {len(data_points)} data points after filtering ndcg@20 > 0")

    if ndcg10_list:
        avg_ndcg10 = sum(ndcg10_list) / len(ndcg10_list)
        print(f"Average NDCG@10 over dataset: {avg_ndcg10:.4f}")
    else:
        print("No samples with ndcg@20 > 0, cannot compute average NDCG@10.")

    # Save to jsonl for convenience
    save_dir = f"data/reranker/raw/{args.split}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.domain_name}.jsonl")

    with open(save_path, "w", encoding="utf-8") as f:
        for dp in data_points:
            f.write(json.dumps(dp, ensure_ascii=False) + "\n")

    print(f"Saved dataset to {save_path}")