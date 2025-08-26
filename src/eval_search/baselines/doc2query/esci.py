import argparse
import json
import os
from tqdm import tqdm
import sys

sys.path.append('./')
from src.eval_search.utils import ndcg_at_k, recall_at_k
from src.Lucene.amazon_c4.search import PyseriniMultiFieldSearch


def merge_results(query_results_list, topk=100):
    merged = {}
    for qres in query_results_list:
        for rank, item in enumerate(qres):
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                docid = item[0]
                score = len(qres) - rank   # higher rank = higher score
            elif isinstance(item, dict) and 'docid' in item:
                docid = item['docid']
                score = item.get('score', len(qres) - rank)
            else:
                raise ValueError(f"Unexpected search result format: {item}")
            merged[docid] = max(merged.get(docid, float('-inf')), score)
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return ranked[:topk]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str,
                        default='results/esci/doc2query/Sports_and_Outdoors/rewrites.json')
    parser.add_argument('--save_path', type=str,
                        default='results/esci/doc2query/metric_results_doc2query.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    search_system = PyseriniMultiFieldSearch(index_dir='database/esci/pyserini_index')
    
    with open(args.res_path, 'r') as f:
        res_dict = json.load(f)

    test_data = []
    for sample_id, value_dict in res_dict.items():
        rewrites = value_dict['rewrites']
        target = value_dict['item_id']
        scores = [1] * len(target)
        test_data.append({'id': sample_id, 'rewrites': rewrites, 'target': target, 'scores': scores})
    
    batch_size = 50
    topk = 100
    results_dict = {}

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i + batch_size]

        for sample in batch:
            sample_id = sample['id']
            rewrites = sample['rewrites']
            targets = sample['target']
            scores = sample['scores']

            # run retrieval for each rewrite
            query_results_list = []
            for q in rewrites:
                qres = search_system.search(q, top_k=topk)  # [(docid, score), ...]
                query_results_list.append(qres)

            # merge results across rewrites
            merged_ranked = merge_results(query_results_list, topk=topk)
            retrieved = [docid for docid, _ in merged_ranked]

            results_dict[sample_id] = {
                'retrieved': str(retrieved),
                'target': str(targets),
                'ndcg@10': ndcg_at_k(retrieved, targets, 10, scores),
                'ndcg@20': ndcg_at_k(retrieved, targets, 20, scores),
                'ndcg@100': ndcg_at_k(retrieved, targets, 100, scores),
                'recall@100': recall_at_k(retrieved, targets, 100),
            }

    # Save metrics
    with open(args.save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Print averages
    ndcg_10 = [v['ndcg@10'] for v in results_dict.values()]
    ndcg_20 = [v['ndcg@20'] for v in results_dict.values()]
    ndcg_100 = [v['ndcg@100'] for v in results_dict.values()]
    recall_100 = [v['recall@100'] for v in results_dict.values()]
    print(f"Average NDCG@10: {sum(ndcg_10) / len(ndcg_10):.4f}")
    print(f"Average NDCG@20: {sum(ndcg_20) / len(ndcg_20):.4f}")
    print(f"Average Recall@100: {sum(recall_100) / len(recall_100):.4f}")
    print(f"Average NDCG@100: {sum(ndcg_100) / len(ndcg_100):.4f}")
