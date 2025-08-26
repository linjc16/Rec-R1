import argparse
import json
import os
import re
from tqdm import tqdm
import pdb

import sys
sys.path.append('./')

from src.eval_search.utils import ndcg_at_k, recall_at_k
from src.Lucene.amazon_c4.search import PyseriniMultiFieldSearch
from src.eval_search.utils import extract_answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='results/esci/gpt-4o_esci_Sports_and_Outdoors.json')
    parser.add_argument('--save_path', type=str, default='results/esci/metric_results_gpt4o.json')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    search_system = PyseriniMultiFieldSearch(index_dir='database/esci/pyserini_index')

    with open(args.res_path, 'r') as f:
        res_dict = json.load(f)

    test_data = []
    for sample_id, value_dict in res_dict.items():
        query = value_dict['generated_text']
        try:
            query = extract_answer(query)
        except:
            query = query

        if isinstance(value_dict['target'], str):
            target = eval(value_dict['target'])  # list of doc ids
        else:
            target = value_dict['target']

        scores = [1] * len(target)
        test_data.append({'id': sample_id, 'query': query, 'target': target, 'scores': scores})

    batch_size = 100
    topk = 100
    results_dict = {}

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i + batch_size]
        queries = [str(item['query']) for item in batch]
        ids = [item['id'] for item in batch]
        targets = {item['id']: item['target'] for item in batch}
        scores = {item['id']: item['scores'] for item in batch}

        
        search_results = search_system.batch_search(queries, top_k=topk, threads=16)

        for idx, sample_id in enumerate(ids):
            query = queries[idx]
            retrieved = [item[0] for item in search_results.get(query, [])]

            results_dict[f"{sample_id}_{query}"] = {
                'id': sample_id,
                'retrieved': str(retrieved),
                'target': str(targets[sample_id]),
                'ndcg@10': ndcg_at_k(retrieved, targets[sample_id], 10, scores[sample_id]),
                'ndcg@20': ndcg_at_k(retrieved, targets[sample_id], 20, scores[sample_id]),
                'ndcg@100': ndcg_at_k(retrieved, targets[sample_id], 100, scores[sample_id]),
                'recall@100': recall_at_k(retrieved, targets[sample_id], 100),
            }
    
    # Save results
    with open(args.save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Print average NDCG
    ndcg_10 = [v['ndcg@10'] for v in results_dict.values()]
    ndcg_20 = [v['ndcg@20'] for v in results_dict.values()]
    ndcg_100 = [v['ndcg@100'] for v in results_dict.values()]
    recall_100 = [v['recall@100'] for v in results_dict.values()]
    print(f"Average NDCG@10: {sum(ndcg_10) / len(ndcg_10):.4f}")
    print(f"Average NDCG@20: {sum(ndcg_20) / len(ndcg_20):.4f}")
    print(f"Average Recall@100: {sum(recall_100) / len(recall_100):.4f}")
    print(f"Average NDCG@100: {sum(ndcg_100) / len(ndcg_100):.4f}")
