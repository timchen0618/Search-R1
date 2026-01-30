import json
import argparse

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def write_last_query(data):
    for item in data:
        item['question'] = item['last_query']
        del item['last_query']
        del item['trajectory']
        del item['search_results']
    return data


def combine_results(data, last_query_search_results, data_type='qampari', topk=100):
    if data_type == 'qampari':
        raw_data = read_jsonl('/scratch/hc3337/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl')
    elif data_type == 'quest':
        raw_data = read_jsonl('/scratch/hc3337/projects/BrowseComp-Plus/data/dev_data_gt_quest.jsonl')
    questions = [item['question_text'] for item in raw_data]
    lens_ = []
    assert len(data) == len(last_query_search_results)
    assert len(data) == len(questions)
    for idx, item in enumerate(data):
        # assert item['query_id'] == idx
        item['question'] = questions[idx]
        contexts = []
        contexts += item['search_results']
        lens_.append(len(contexts))
        contexts += last_query_search_results[idx]['ctxs']
        
        contexts = contexts[:topk]
        
        item['ctxs'] = contexts
        del item['last_query']
        del item['trajectory']
        del item['search_results']
    print(f"Average length of contexts: {sum(lens_) / len(lens_)}")
    print(f"Max length of contexts: {max(lens_)}")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='output_infly.jsonl')
    parser.add_argument('--last_query_search_results', type=str, default='output_infly.jsonl')
    parser.add_argument('--output_file', type=str, default='ooo.jsonl')
    parser.add_argument('--command', '-c', type=str, default='last_query', choices=['last_query', 'combine_results'])
    parser.add_argument('--data_type', type=str, default='qampari', choices=['qampari', 'quest'])
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()

    data = read_jsonl(args.input_file)
    if args.command == 'last_query':
        data = write_last_query(data)
    elif args.command == 'combine_results':
        last_query_search_results = read_jsonl(args.last_query_search_results)
        data = combine_results(data, last_query_search_results, args.data_type, args.topk)

    write_jsonl(args.output_file, data)

if __name__ == '__main__':
    main()
    
    # generate results for qampari
    
    # python generate_qampari_results.py --input_file outputs_qampari/output_qwen3-0.6b.jsonl --output_file outputs_qampari/last_query/output_qwen3-0.6b.jsonl
    # python generate_qampari_results.py --input_file outputs_qampari/output_contriever.jsonl --output_file outputs_qampari/last_query/output_contriever.jsonl
    # python generate_qampari_results.py --input_file outputs_qampari/output_infly.jsonl --output_file outputs_qampari/last_query/output_infly.jsonl
    # python generate_qampari_results.py --input_file outputs_qampari/output_qwen3-0.6b_finetuned.jsonl --output_file outputs_qampari/last_query/output_qwen3-0.6b_finetuned.jsonl
    # python generate_qampari_results.py --input_file outputs_qampari/output_contriever_finetuned.jsonl --output_file outputs_qampari/last_query/output_contriever_finetuned.jsonl
    # python generate_qampari_results.py --input_file outputs_qampari/output_infly_finetuned.jsonl --output_file outputs_qampari/last_query/output_infly_finetuned.jsonl
    
    # python generate_qampari_results.py --input_file outputs_quest/output_qwen3-0.6b_quest.jsonl --output_file outputs_quest/last_query/output_qwen3-0.6b_quest.jsonl
    # python generate_qampari_results.py --input_file outputs_quest/output_contriever_quest.jsonl --output_file outputs_quest/last_query/output_contriever_quest.jsonl
    # python generate_qampari_results.py --input_file outputs_quest/output_infly_quest.jsonl --output_file outputs_quest/last_query/output_infly_quest.jsonl
    
    # python generate_qampari_results.py --input_file outputs_qampari/output_qwen3-0.6b.jsonl --last_query_search_results /scratch/hc3337/projects/autoregressive/results/base_retrievers/qwen3-0.6b/qampari_last_queries_searchr1.jsonl --command combine_results --topk 100 --output_file /scratch/hc3337/projects/autoregressive/results/base_retrievers/qwen3-0.6b/qampari_agentic_final_results_topk100_searchr1.jsonl
    # python generate_qampari_results.py --input_file outputs_qampari/output_contriever.jsonl --last_query_search_results /scratch/hc3337/projects/autoregressive/results/base_retrievers/contriever/qampari_last_queries_searchr1.jsonl --command combine_results --topk 100 --output_file /scratch/hc3337/projects/autoregressive/results/base_retrievers/contriever/qampari_agentic_final_results_topk100_searchr1.jsonl
    # python generate_qampari_results.py --input_file outputs_qampari/output_infly.jsonl --last_query_search_results /scratch/hc3337/projects/autoregressive/results/base_retrievers/infly/qampari_last_queries_searchr1.jsonl --command combine_results --topk 100 --output_file /scratch/hc3337/projects/autoregressive/results/base_retrievers/infly/qampari_agentic_final_results_topk100_searchr1.jsonl
    
    
    # python generate_qampari_results.py --input_file outputs_quest/output_qwen3-0.6b_quest.jsonl --last_query_search_results /scratch/hc3337/projects/autoregressive/results/base_retrievers/qwen3-0.6b/quest_last_queries_searchr1.jsonl --command combine_results --topk 100 --output_file /scratch/hc3337/projects/autoregressive/results/base_retrievers/qwen3-0.6b/quest_agentic_final_results_topk100_searchr1.jsonl --data_type quest
    # python generate_qampari_results.py --input_file outputs_quest/output_contriever_quest.jsonl --last_query_search_results /scratch/hc3337/projects/autoregressive/results/base_retrievers/contriever/quest_last_queries_searchr1.jsonl --command combine_results --topk 100 --output_file /scratch/hc3337/projects/autoregressive/results/base_retrievers/contriever/quest_agentic_final_results_topk100_searchr1.jsonl --data_type quest
    # python generate_qampari_results.py --input_file outputs_quest/output_infly_quest.jsonl --last_query_search_results /scratch/hc3337/projects/autoregressive/results/base_retrievers/infly/quest_last_queries_searchr1.jsonl --command combine_results --topk 100 --output_file /scratch/hc3337/projects/autoregressive/results/base_retrievers/infly/quest_agentic_final_results_topk100_searchr1.jsonl --data_type quest