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
    topks = []
    for item in data:
        trajectory = item['trajectory']
        cur_subqueries, cur_topks = _parse_search_tag(trajectory)
        subqueries.extend(cur_subqueries)
        topks.extend(cur_topks)
        num_queries_per_inst.append(len(cur_subqueries))
    return subqueries, topks, num_queries_per_inst


def _parse_search_tag(text: str) -> Tuple[Optional[str], Optional[int]]:
    pattern = re.compile(r"<search(?:\s+topk=(\d+))?\s*>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        print('No matches found')
        return [], []

    subqueries = []
    topks = []
    print('-----')
    for _match in matches:
        print(_match)
        topk_str, query = _match
        topk = int(topk_str) if topk_str else None
        query = query.strip()
        if query == 'query':
            continue
        subqueries.append(query)
        topks.append(topk)
    return subqueries, topks


def perform_retrieval(subqueries, retrieval_manager, retrieval_batch_size):
    all_search_results = []
    for i in range(0, len(subqueries), retrieval_batch_size):
        batch_subqueries = subqueries[i:i+retrieval_batch_size]
        search_results = retrieval_manager.batch_search(batch_subqueries)
        all_search_results.extend(search_results)
    return all_search_results



data = read_jsonl('verl_checkpoints/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo_inference_musique_sm/test_outputs.jsonl')[:5]
port = 8000
TOPK = 50
retrieval_batch_size = 512





subqueries, topks, num_queries_per_inst = collect_subqueries(data)

print('Start Printing Subqueries and Topks')

print('subqueries:', subqueries)
print('topks:', topks)
print('num_queries_per_inst:', num_queries_per_inst)

# # perform retrieval batch by batch
print('===============================================')
print('Start Performing Retrieval')
print('===============================================')
retrieval_manager = RetrievalManager(search_url=f'http://127.0.0.1:{port}/search', topk=TOPK)
all_search_results = perform_retrieval(subqueries, retrieval_manager, retrieval_batch_size)
print('all_search_results[0]:', all_search_results[0])
print('===============================================')
print('End Performing Retrieval')
print('===============================================')

# write_jsonl('test_outputs_with_search_results.jsonl', data)