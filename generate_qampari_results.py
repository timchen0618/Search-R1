import json
import argparse

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='output_infly.jsonl')
    parser.add_argument('--output_file', type=str, default='output_infly_last_query.jsonl')
    args = parser.parse_args()

    data = read_jsonl(args.input_file)
    for item in data:
        item['question'] = item['last_query']
        del item['last_query']
        del item['trajectory']
        del item['search_results']

    write_jsonl(args.output_file, data)

if __name__ == '__main__':
    main()
    
    # generate results for qampari
    
    # python generate_qampari_results.py --input_file outputs_qampari/output_qwen3-0.6b.jsonl --output_file outputs_qampari/last_query/output_qwen3-0.6b.jsonl
    # python generate_qampari_results.py --input_file outputs_qampari/output_contriever.jsonl --output_file outputs_qampari/last_query/output_contriever.jsonl
    # python generate_qampari_results.py --input_file outputs_qampari/output_infly.jsonl --output_file outputs_qampari/last_query/output_infly.jsonl