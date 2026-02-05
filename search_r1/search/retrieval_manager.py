import requests
from typing import List, Tuple



class RetrievalManager:
    def __init__(
        self,
        search_url: str,
        topk: int,
    ):
        self.search_url = search_url
        self.topk = topk

    def batch_search(self, queries: List[str] = None) -> Tuple[List[str], float]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        if len(queries) == 0 or (queries is None):
            return []
        
        rrr = self._batch_search(queries)
        print('-----------=============================-------------------------')
        print(rrr)
        results = rrr['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, query_texts):
        payload = {
            "queries": query_texts,
            "topk": [int(self.topk)] * len(query_texts),
            "return_scores": True
        }
        return requests.post(self.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        contexts = []
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            contexts.append({
                "id": doc_item['document']['id'],
                "title": title,
                "text": text
            })

        return contexts
