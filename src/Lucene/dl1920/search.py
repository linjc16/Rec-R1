import json
from pyserini.search.lucene import LuceneSearcher
import time
import re
import pdb



class PyseriniMultiFieldSearch:
    def __init__(self, index_dir="pyserini_index"):
        """Initialize Pyserini MultiField Searcher"""
        self.searcher = LuceneSearcher(index_dir)
        self.searcher.set_bm25(1.2, 0.75)  # Set BM25 scoring for ranking

    def search(self, query_str, top_k=10):
        """Perform search across multiple fields"""
        
        # Construct a query that searches across multiple fields
        # query = f"title:{query_str} OR store:{query_str} OR details:{query_str} OR main_category:{query_str}"
        
        query = f"contents:{query_str}"

        # Execute the search
        hits = self.searcher.search(query, k=top_k)

        results = []
        for hit in hits:
            doc = json.loads(hit.raw)  # Get raw document
            results.append((doc["id"], doc["title"], hit.score))  # (parent_asin, title, relevance score)

        return results


    def batch_search(self, queries, top_k=10, threads=4):
        """
        Perform parallel search across multiple fields using batch_search
        :param queries: List of query strings
        :param top_k: Number of results per query
        :param threads: Number of parallel threads for searching
        :return: Dictionary {query: [(parent_asin, title, score), ...]}
        """
        # Construct field-specific queries
        # field_queries = [
        #     f"(title:{query} OR store:{query} OR details:{query} OR main_category:{query}"
        #     for query in queries
        # ]
        # contents
        field_queries = [
            f"(contents:{query})"
            for query in queries
        ]
        
        # Perform batch search in parallel
        results_dict = self.searcher.batch_search(
            field_queries,  # List of queries
            [str(i) for i in range(len(queries))],  # Unique query IDs
            k=top_k,
            threads=threads  # Enable parallel searching
        )
        
        # Format results as {query: [(parent_asin, title, score), ...]}
        final_results = {}
        for i, query in enumerate(queries):
            hits = results_dict[str(i)]  # Get results for query `i`
            formatted_results = [
                (json.loads(hit.raw)["id"], json.loads(hit.raw)["contents"], hit.score)
                for hit in hits
            ]
            final_results[query] = formatted_results

        return final_results

# Example Usage
if __name__ == "__main__":
    search_system = PyseriniMultiFieldSearch(index_dir='database/dl1920/pyserini_index')
    
    queries = [
        "3-Pack Replacement for Whirlpool AND (Water Inlet OR brand new)",
    ]
    
    tic = time.time()
    search_results = search_system.batch_search(queries, top_k=3, threads=32)
    print(f"Search time: {time.time() - tic:.2f}s")
    # Print results
    for query, results in search_results.items():
        # print(f"\n🔍 Query: {query}")
        for asin, content, score in results:
            print(f"  ASIN: {asin}, Content: {content}, Score: {score}")
