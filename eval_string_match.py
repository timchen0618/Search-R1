import json
import re
from tqdm import tqdm
from verl.utils.reward_score.qa_em import extract_solution, compute_score_em
from datasets import load_dataset

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

def parse_output(file_path, ground_truths, _print=False):
    data = read_jsonl(file_path)
    assert len(ground_truths) == len(data), f'Length mismatch: {len(ground_truths)} != {len(data)}'
    num_search_queries = []
    answers = []
    scores = []
    for idx, item in enumerate(tqdm(data)):
        trajectory = item['trajectory']
        # print(trajectory)
        # Extract all <search>...</search> tags
        search_pattern = re.compile(r'<search>(.*?)</search>', re.DOTALL)
        search_matches = search_pattern.findall(trajectory)
        
        if _print:
            print(f'\n{"="*100}')
            print(f'Entry {idx}:')
            print(f'{"="*100}\n')
        
        if search_matches:
            for i, search_query in enumerate(search_matches, 1):
                if search_query.strip() == 'query':
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
    print(f'Average score: {sum(scores) / len(scores)}')
    


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
    ground_truths = extract_data('nq')
    
    parse_output('output_base_topk3_obs2048_bsz16.jsonl', ground_truths)