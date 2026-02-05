import json
from search_r1.search.retrieval_manager import RetrievalManager
from typing import Optional, Tuple
import re

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def collect_subqueries(data):
    subqueries = []
    num_queries_per_inst = []
    for item in data:
        trajectory = item['trajectory']
        subqueries.append(trajectory)
    return subqueries, num_queries_per_inst


def _parse_search_tag(text: str) -> Tuple[Optional[str], Optional[int]]:
    pattern = re.compile(r"<search(?:\s+topk=(\d+))?\s*>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return [], []

    print('-----')
    for _match in matches:
        print(_match)
        topk_str, query = _match
        topk = int(topk_str) if topk_str else None
        query = query.strip()
        if query == 'query':
            continue
        subqueries.append(query)
        num_queries_per_inst.append(topk)
    return query.strip(), topk


def perform_retrieval(data, retrieval_manager, retrieval_batch_size):
    for i in range(0, len(data), retrieval_batch_size):
        batch_data = data[i:i+retrieval_batch_size]
        batch_subqueries = [item['trajectory'] for item in batch_data]
        search_results = retrieval_manager.batch_search(batch_subqueries)
        for j in range(len(batch_data)):
            batch_data[j]['search_results'] = search_results[j]
    return data



data = read_jsonl('verl_checkpoints/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo_inference_musique_sm/test_outputs.jsonl')
port = 8000
TOPK = 50
retrieval_batch_size = 512



# retrieval_manager = RetrievalManager(search_url=f'http://127.0.0.1:{port}/search', topk=TOPK)

# subqueries, num_queries_per_inst = collect_subqueries(data)

# # perform retrieval batch by batch
# data = perform_retrieval(data, retrieval_manager, retrieval_batch_size)

# write_jsonl('test_outputs_with_search_results.jsonl', data)