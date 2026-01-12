import json
import re

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def parse_output(file_path):
    data = read_jsonl(file_path)
    for idx, item in enumerate(data, 1):
        trajectory = item['trajectory']
        print(trajectory)
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
        else:
            print('No <search> tags found.')
        
        print(f'{"-"*100}')

if __name__ == '__main__':
    parse_output('output.jsonl')