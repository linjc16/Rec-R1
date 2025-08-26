import argparse
import json
import os
import re
from tqdm import tqdm
import pdb

import sys
sys.path.append('./')

from src.eval_search.utils import ndcg_at_k, recall_at_k
from src.Lucene.esci.search import PyseriniMultiFieldSearch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, choices=['Video_Games', 'Baby_Products', 'Office_Products', 'Sports_and_Outdoors'], default='Video_Games')
    parser.add_argument('--test_data_dir', type=str, default='data/esci/test_subset')
    parser.add_argument('--save_path', type=str, default='results/esci/metric_res/ori_query.json')
    args = parser.parse_args()
    
    search_system = PyseriniMultiFieldSearch(index_dir='database/esci/pyserini_index')
    
    # Load the test data
    test_data_path = os.path.join(args.test_data_dir, f"{args.domain}.json")
    with open(test_data_path, "r") as f:
        raw_test_data = json.load(f)
    

    test_data = []
    for idx, entry in enumerate(raw_test_data):
        query = entry['query']
        target = entry['item_id']
        scores = [1] * len(target)
        test_data.append({'id': idx, 'query': query, 'target': target, 'scores': scores})
    
    
    ndcg = []
    batch_size = 100
    topk = 100
    results_dict = {}
    
    
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i + batch_size]
        queries = [item['query'] for item in batch]
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