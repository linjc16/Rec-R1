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


def extract_answer(generated_text):
    # extract from \nAssistant:
    # try:
    #     generated_text = generated_text.split("\nAssistant:")[1]
    # except:
    #     generated_text = generated_text.split("\nassistant:")[1]
    
    # findall <answer> </answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, generated_text, re.DOTALL)  # Use re.DOTALL to match multiline content

    if len(matches) > 0:
        generated_text = matches[-1]
        try:
            # json.loads(generated_text)
            generated_text = json.loads(generated_text)['query']
        except:
            generated_text = matches[-1]

    return generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='results/Baby/amazon_c4/eval_results_rec-r1.json')
    args = parser.parse_args()

    search_system = PyseriniMultiFieldSearch(index_dir='database/amazon_c4/pyserini_index')

    with open(args.res_path, 'r') as f:
        res_dict = json.load(f)

    
    test_data = []
    for sample_id, value_dict in res_dict.items():
        query = str(value_dict['generated_text'])
        try:
            query = extract_answer(query)
        except:
            query = query
        query = str(query)
        target = value_dict['target']
        test_data.append({'id': sample_id, 'query': query, 'target': target})
    
    batch_size = 100
    topk = 100
    results_dict = {}
    
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [str(item['query']) for item in batch]
        ids = [item['id'] for item in batch]
        targets = {item['id']: item['target'] for item in batch}
        
        search_results = search_system.batch_search(queries, top_k=topk, threads=16)
        
        for idx, sample_id in enumerate(ids):
            query = queries[idx]
            retrieved = [item[0] for item in search_results.get(query, [])]
            
            results_dict[f"{sample_id}_{query}"] = {
                'id': sample_id,
                'retrieved': str(retrieved),
                'target': str(targets[sample_id]),
                'ndcg@10': ndcg_at_k(retrieved, targets[sample_id], 10),
                'ndcg@20': ndcg_at_k(retrieved, targets[sample_id], 20),
                'ndcg@100': ndcg_at_k(retrieved, targets[sample_id], 100),
                'recall@100': recall_at_k(retrieved, [targets[sample_id]], 100),
            }

    # Print average NDCG
    ndcg_10 = [v['ndcg@10'] for v in results_dict.values()]
    ndcg_20 = [v['ndcg@20'] for v in results_dict.values()]
    ndcg_100 = [v['ndcg@100'] for v in results_dict.values()]
    recall_100 = [v['recall@100'] for v in results_dict.values()]
    print(f"Average NDCG@10: {sum(ndcg_10) / len(ndcg_10):.4f}")
    print(f"Average NDCG@20: {sum(ndcg_20) / len(ndcg_20):.4f}")
    print(f"Average Recall@100: {sum(recall_100) / len(recall_100):.4f}")
    print(f"Average NDCG@100: {sum(ndcg_100) / len(ndcg_100):.4f}")

