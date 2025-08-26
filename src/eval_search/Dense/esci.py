import json
import sys
import os
from tqdm import tqdm
import pdb
sys.path.append('./')

from src.Dense.esci.search import FaissHNSWSearcher
from src.eval_search.utils import ndcg_at_k, recall_at_k
import argparse

from src.eval_search.utils import extract_answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['roberta-base', 'blair-base', 'blair-large', 'roberta-large', 'simcse-base', 'simcse-large'], default='roberta-base')
    parser.add_argument('--domain', type=str, choices=['Video_Games', 'Baby_Products', 'Office_Products', 'Sports_and_Outdoors'], default='Video_Games')
    parser.add_argument('--test_data_dir', type=str, default=None)
    parser.add_argument('--test_file_path', type=str, default=None)
    args = parser.parse_args()
    

    if args.model_name == 'roberta-base':
        model_path = "FacebookAI/roberta-base"
        index_path = f"data/esci/raw/dense_index/roberta-base/faiss_hnsw_index.bin"
        doc_ids_path = f"data/esci/raw/esci/doc_ids.npy"
    elif args.model_name == 'roberta-large':
        model_path = "FacebookAI/roberta-large"
        index_path = f"data/esci/raw/dense_index/roberta-large/faiss_hnsw_index.bin"
        doc_ids_path = f"data/esci/raw/esci/doc_ids.npy"
    elif args.model_name == 'blair-base':
        model_path = "hyp1231/blair-roberta-base"
        index_path = f"data/esci/raw/dense_index/blair-base/faiss_hnsw_index.bin"
        doc_ids_path = f"data/esci/raw/esci/doc_ids.npy"
    elif args.model_name == 'blair-large':
        model_path = "hyp1231/blair-roberta-large"
        index_path = f"data/esci/raw/dense_index/blair-large/faiss_hnsw_index.bin"
        doc_ids_path = f"data/esci/raw/esci/doc_ids.npy"
    elif args.model_name == 'simcse-base':
        model_path = 'princeton-nlp/sup-simcse-roberta-base'
        index_path = f"data/esci/raw/dense_index/simcse-base/faiss_hnsw_index.bin"
        doc_ids_path = f"data/esci/raw/esci/doc_ids.npy"
    elif args.model_name == 'simcse-large':
        model_path = 'princeton-nlp/sup-simcse-roberta-large'
        index_path = f"data/esci/raw/dense_index/simcse-large/faiss_hnsw_index.bin"
        doc_ids_path = f"data/esci/raw/esci/doc_ids.npy"
    else:
        raise NotImplementedError('Model not supported')


    search_system = FaissHNSWSearcher(model_name=model_path, 
                                 index_path=index_path, 
                                 doc_ids_path=doc_ids_path)
    
    
    if args.test_data_dir is not None:
        # Load the test data
        test_data_path = os.path.join(args.test_data_dir, f"{args.domain}.json")
        with open(test_data_path, "r") as f:
            raw_test_data = json.load(f)
        

        test_data = []
        for i, entry in enumerate(raw_test_data):
            query = entry['query']
            target = entry['item_id']
            scores = [1] * len(target)
            test_data.append({'id': str(i), 'query': query, 'target': target, 'scores': scores})
    elif args.test_file_path is not None:
        with open(args.test_file_path, "r") as f:
            raw_test_data = json.load(f)
        
        test_data = []
        for key, entry in raw_test_data.items():
            query = extract_answer(str(entry['generated_text']))
            target = eval(entry['target'])
            scores = [1] * len(target)
            test_data.append({'id': str(key), 'query': query, 'target': target, 'scores': scores})
    else:
        raise ValueError("Either test_data_dir or test_file_path must be provided")

    
    ndcg = []
    batch_size = 32
    topk = 100
    results_dict = {}

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [item['query'] for item in batch]
        ids = [item['id'] for item in batch]
        targets = {item['id']: item['target'] for item in batch}
        scores = {item['id']: item['scores'] for item in batch}

        results = search_system.batch_search(queries, top_k=topk, threads=16)

        for idx, query in enumerate(queries):
            sample_id = ids[idx]
            retrieved = [result[0] for result in results.get(query, [])]

            ndcg_score = ndcg_at_k(retrieved, targets[sample_id], 10, scores[sample_id])
            ndcg.append(ndcg_score)

            results_dict[f"{sample_id}_{query}"] = {
                'id': sample_id,
                'retrieved': str(retrieved),
                'target': str(targets[sample_id]),
                'ndcg@10': ndcg_score,
                'ndcg@20': ndcg_at_k(retrieved, targets[sample_id], 20, scores[sample_id]),
                'ndcg@100': ndcg_at_k(retrieved, targets[sample_id], 100, scores[sample_id]),
                'recall@100': recall_at_k(retrieved, targets[sample_id], 100),
            }

    # with open(args.save_path, 'w') as f:
    #     json.dump(results_dict, f, indent=2)

    # Print average NDCG
    ndcg_10 = [v['ndcg@10'] for v in results_dict.values()]
    ndcg_20 = [v['ndcg@20'] for v in results_dict.values()]
    ndcg_100 = [v['ndcg@100'] for v in results_dict.values()]
    recall_100 = [v['recall@100'] for v in results_dict.values()]
    print(f"Average NDCG@10: {sum(ndcg_10) / len(ndcg_10):.4f}")
    print(f"Average NDCG@20: {sum(ndcg_20) / len(ndcg_20):.4f}")
    print(f"Average Recall@100: {sum(recall_100) / len(recall_100):.4f}")
    print(f"Average NDCG@100: {sum(ndcg_100) / len(ndcg_100):.4f}")