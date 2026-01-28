import json
import argparse

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def compute_inference_time(data):
    total_inference_time = 0.0
    total_retrieval_time = 0.0
    for item in data:
        total_inference_time += item['total_inference_time']
        total_retrieval_time += item['total_retrieval_time']
    return total_inference_time, total_retrieval_time

def compute_number_of_calls(data):
    total_number_of_calls = []
    for item in data:
        total_number_of_calls.append(item['trajectory'].split('<answer> Beijing </answer>')[-1].count('<search>'))
    return sum(total_number_of_calls) / len(total_number_of_calls)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='output_infly.jsonl')
    args = parser.parse_args()

    data = read_jsonl(args.input_file)[:200]
    
    print(f"Computing inference time for {len(data)} questions")
    try:
        total_inference_time, total_retrieval_time = compute_inference_time(data)
        print(f"Total inference time: {total_inference_time} seconds")
        print(f"Total retrieval time: {total_retrieval_time} seconds")
        
        print(f"Average inference time: {total_inference_time / len(data)} seconds")
        print(f"Average retrieval time: {total_retrieval_time / len(data)} seconds")
    except:
        print(f"Error computing inference time for {args.input_file}")
        
    try:
        number_of_calls = compute_number_of_calls(data)
        print(f"Average number of calls: {number_of_calls}")
    except:
        print(f"Error computing number of calls for {args.input_file}")
        
    print(data[0]['trajectory'])

if __name__ == '__main__':
    main()