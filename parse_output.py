import json
import re

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def parse_output(file_path):
    data = read_jsonl(file_path)
    num_search_queries = []
    for idx, item in enumerate(data, 1):
        trajectory = item['trajectory']
        # print(trajectory)
        # Extract all <search>...</search> tags
        search_pattern = re.compile(r'<search>(.*?)</search>', re.DOTALL)
        search_matches = search_pattern.findall(trajectory)
        
        print(f'\n{"="*100}')
        print(f'Entry {idx}:')
        print(f'{"="*100}\n')
        
        if search_matches:
            for i, search_query in enumerate(search_matches, 1):
                if search_query.strip() == 'query':
                    continue
                print(f'Search {i-1}:')
                print(f'<search>{search_query.strip()}</search>')
                print()
            num_search_queries.append(len(search_matches)-1)
        else:
            print('No <search> tags found.')
            num_search_queries.append(0)
        
        print(f'{"-"*100}')

    print(f'Number of search queries: {sum(num_search_queries) / len(num_search_queries)}')

if __name__ == '__main__':
    parse_output('output_base_topk3_obs2048_bsz16.jsonl')