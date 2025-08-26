import json
import sys
import os
from tqdm import tqdm
import pdb
sys.path.append('./')

from src.Lucene.dl1920.search import PyseriniMultiFieldSearch
from src.eval_search.utils import ndcg_at_k, recall_at_k

import argparse

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['dl19', 'dl20'], help='Dataset to use (dl19 or dl20)', default='dl19')
    args = parser.parse_args()
    search_system = PyseriniMultiFieldSearch(index_dir="database/dl1920/pyserini_index")
    
    with open(f"data/{args.dataset}/raw/qrels/test.tsv", "r", encoding="utf-8") as file:
        qrel_test = parse_qrel(file.readlines())
        # qrel_test = [line.strip().split("\t") for line in file]

    # qrel_test = qrel_test[1:]  # remove the header
    
        
    # read code/data/raw_data/fever/queries.jsonl
    with open(f"data/{args.dataset}/raw/queries.jsonl", "r", encoding="utf-8") as file:
        queries = [json.loads(line) for line in file]
    queries_dict = {q['_id']: q['text'] for q in queries}
    
    test_data = []
    for qid, value in qrel_test.items():
        test_data.append({
            "qid": qid,
            'query': queries_dict[qid],
            "target": value['targets'],
            "score": value['scores']
        })

    ndcg_10_list = []
    ndcg_20_list = []
    ndcg_100_list = []
    recall_100_list = []
    # ========================

    batch_size = 100
    topk = 100 
    
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [item['query'] for item in batch]
        targets = {item['query']: item['target'] for item in batch} 
        scores = {item['query']: item['score'] for item in batch}
        
        results = search_system.batch_search(queries, top_k=topk, threads=16)
        
        for query in queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg_10_list.append(ndcg_at_k(retrieved, targets[query], 10, rel_scores=scores[query]))
            ndcg_20_list.append(ndcg_at_k(retrieved, targets[query], 20, rel_scores=scores[query]))
            ndcg_100_list.append(ndcg_at_k(retrieved, targets[query], 100, rel_scores=scores[query]))
            recall_100_list.append(recall_at_k(retrieved, targets[query], 100))

    print(f"Average NDCG@10:  {sum(ndcg_10_list) / len(ndcg_10_list):.4f}")
    print(f"Average NDCG@20:  {sum(ndcg_20_list) / len(ndcg_20_list):.4f}")
    print(f"Average NDCG@100: {sum(ndcg_100_list) / len(ndcg_100_list):.4f}")
    print(f"Average Recall@100: {sum(recall_100_list) / len(recall_100_list):.4f}")
