import csv
import json
from tqdm import tqdm
def read_tsv(file_path):
    # id	text	title
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        return [row for row in reader]

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            
            
if __name__ == '__main__':
    tsv_file = '/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv'
    jsonl_file = '/scratch/hc3337/qampari_searchr1/qampari_corpus.jsonl'
    data = read_tsv(tsv_file)
    new_data = []
    for item in tqdm(data):
        if item[0] == 'id':
            continue
        title = item[2]
        text = item[1]
        new_data.append({
            'id': item[0],
            'contents': '"' + title + '"\n' + text,
        })
    
    write_jsonl(jsonl_file, new_data)