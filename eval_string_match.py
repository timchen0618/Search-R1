import json
import re
from tqdm import tqdm
from verl.utils.reward_score.qa_em import extract_solution, compute_score_em
from datasets import load_dataset
import argparse


def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    
    
def extract_question(trajectory):
    question_pattern = re.compile(r'Question:(.*?)\n\n\n', re.DOTALL)
    matches = question_pattern.findall(trajectory)
    if matches:
        return matches[0].strip()
    else:
        return ""

def parse_output(file_path, ground_truths, _print=False, num_examples=None):
    data = read_jsonl(file_path)
    # assert len(ground_truths) == len(data), f'Length mismatch: {len(ground_truths)} != {len(data)}'\
    if len(ground_truths) != len(data):
        print(f'[Warning] Length mismatch: {len(ground_truths)} != {len(data)}')
        
    num_search_queries = []
    answers = []
    scores = []
    if num_examples is not None:
        data = data[:num_examples]
    for idx, item in enumerate(tqdm(data)):
        trajectory = item['trajectory']
        # print(trajectory)
        # Extract all <search>...</search> tags
        # pattern = r'<(?P<action>search)(?:\s+topk=(?P<topk>\d+))?>(?P<content>.*?)</(?P=action)>'
        search_pattern = re.compile(r'<search>(.*?)</search>|<(?P<action>search)(?:\s+topk=(?P<topk>\d+))?>(?P<content>.*?)</(?P=action)>', re.DOTALL)
        search_matches = search_pattern.findall(trajectory)
        
        if _print:
            print(f'\n{"="*100}')
            print(f'Entry {idx}:')
            print(f'{"="*100}\n')
        
        if search_matches:
            for i, search_query in enumerate(search_matches, 1):
                if isinstance(search_query, tuple):
                    search_query = search_query[2].strip()
                elif isinstance(search_query, str):
                    search_query = search_query.strip()
                
                if search_query == 'query':
                    continue
                
                if _print:
                    print(f'Search {i-1}:')
                    print(f'<search>{search_query.strip()}</search>')
                    print()
            num_search_queries.append(len(search_matches)-1)
        else:
            if _print:
                print('No <search> tags found.')
            num_search_queries.append(0)
        
        if _print:
            print(f'{"-"*100}')
        
        prediction = extract_solution(trajectory)
        answer_list = ground_truths[idx]['golden_answers']
        
        ## Check question mismatch
        data_question = ground_truths[idx]['question']
        question = extract_question(trajectory)
        assert question.strip('?') == data_question.strip('?'), f'Question mismatch: {question} != {data_question}'

        # Compute scores
        inst_scores = [compute_score_em(trajectory, {"target": gold_answer}) for gold_answer in answer_list]
        scores.append(max(inst_scores))
        answers.append(prediction)
        
        ## Print results
        if _print:
            print(f'Prediction: {prediction}')
            print(f'Answer list: {answer_list}')
            print(f'Question: {question}')
            print(f'Data question: {data_question}')
            print(f'Instance scores: {inst_scores}')
        
    print(f'Number of search queries: {sum(num_search_queries) / len(num_search_queries)}')
    print(f'Average score: {100*sum(scores) / len(scores)}')
    


def extract_data(data_path):
    if data_path in ['nq', 'musique', 'hotpotqa', '2wikimultihopqa', 'bamboogle']:
        dataset = load_dataset('RUC-NLPIR/FlashRAG_datasets', data_path)
        if 'test' in dataset:
            for item in dataset['test']:
                print(item.keys())
                break
            return dataset['test']
        else:
            return dataset['dev']
    else:
        return read_jsonl(data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, default='nq')
    parser.add_argument('--output_path', '-o', type=str, default='output_base_topk3_obs2048_bsz16.jsonl')
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--num_examples', '-n', type=int, default=None)
    args = parser.parse_args()

    ground_truths = extract_data(args.data_path)
    parse_output(args.output_path, ground_truths, args.print, args.num_examples)