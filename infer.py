import transformers
import torch
from tqdm import tqdm
import random
from datasets import load_dataset
import requests
import json
import argparse
import os



# /scratch/hc3337/projects/diverse_response/data/dev_data_qampari_corpus.jsonl

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def append_to_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + '\n')

# question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"



# Multi Answer
# def prepare_prompt(question):
#     return f"""Answer the given question. Each question has multiple answers. You should provide all the answers. \
#     You must conduct reasoning inside <think> and </think> first every time you get new information. \
#     After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
#     You can search as many times as your want. \
#     If you find one answer from the documents, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. You should provide all the answers, so you can output <answer> and </answer> multiple times. For example, <answer> Beijing </answer>. Question: {question}\n"""

    
def prepare_prompt(question, template_type='dynamic'):
    if template_type == 'dynamic':
        return f"""Answer the given question. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some knowledge, you can call a search engine by <search topk=N> query </search> and it will return the top-N searched results between <information> and </information>. Please always specify the topk value, which is an integer between 1 and 10. \
        You can search as many times as your want. \
        If you find one answer from the documents, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    elif template_type == 'base':
        return f"""Answer the given question. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
        You can search as many times as your want. \
        If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# # Prepare the message
# prompt = f"""Answer the given question. \
# You must conduct reasoning inside <think> and </think> first every time you get new information. \
# After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
# You can search as many times as your want. \
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""



# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

# def get_query(text):
#     import re
#     pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
#     matches = pattern.findall(text)
#     if matches:
#         return matches[-1]
#     else:
#         return None
    
    
# def get_query(text):
#     import re
    
#     pattern = r'<(?P<action>search|answer)(?:\s+topk=(?P<topk>\d+))?>(?P<content>.*?)</(?P=action)>'
#     # pattern = r'<(search|answer)>(.*?)</\1>'
#     match = re.search(pattern, text, re.DOTALL)
#     # if match:
#     #     content = match.group(2).strip()  # Return only the content inside the tags
#     #     action = match.group(1)
#     if match:
#         content = match.group('content').strip()
#         action = match.group('action')
#         topk = match.group('topk')
#     else:
#         content = ''
#         action = None
#         topk = None
        
#     print(f'TEXT: {text}, action: {action}, topk: {topk}, content: {content}')
#     if action == 'search':
#         try:
#             topk = int(topk)
#         except:
#             topk = None 
#         return content, topk
#     else:
#         return None, None

def get_query(text):
    import re
    
    pattern = r'<(?P<action>search)(?:\s+topk=(?P<topk>\d+))?>(?P<content>.*?)</(?P=action)>'

    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        match = matches[-1]
        print(f'MATCH: {match} | length: {len(matches)}')
        if match[1].isdigit():
            return match[2].strip(), int(match[1])
        else:
            return match[2].strip(), None
    else:
        return None, None

def search(query: str, topk: int, port: int):
    payload = {
            "queries": [query],
            "topk": [topk],
            "return_scores": True
        }
    results = requests.post(f"http://127.0.0.1:{port}/retrieve", json=payload).json()['result']
    return results[0]

def _passages2string(retrieval_result):
    format_reference = ''
    output_dicts = []
    for idx, doc_item in enumerate(retrieval_result):               
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        output_dicts.append({
            "id": doc_item['document']['id'],
            "title": title,
            "text": text
        })
    return format_reference, output_dicts

def main(args):
    # Time tracking
    import time
      
    print("URL: ", f"http://127.0.0.1:{args.port}/retrieve")
    
    # Model ID and device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    # Initialize the tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    # Get Data
    if args.data_path in ['nq', 'musique', 'hotpotqa', '2wikimultihopqa', 'bamboogle']:
        from datasets import load_dataset
        dataset = load_dataset('RUC-NLPIR/FlashRAG_datasets', args.data_path)
        if 'test' in dataset:
            questions = [item['question'] for item in dataset['test']]
        else:
            questions = [item['question'] for item in dataset['dev']]
    else:
        raw_data = read_jsonl(args.data_path)
        questions = [item['question_text'] if 'question_text' in item else item['query'] for item in raw_data]
        if os.path.exists(args.output_file):
            questions = questions[len(read_jsonl(args.output_file)):]
            print(f"Skipping {len(read_jsonl(args.output_file))} questions")
        else:
            questions = questions


    # Inference Loop
    for q_idx, question in enumerate(tqdm(questions)):
        total_retrieval_time = 0.0
        total_inference_time = 0.0  
    
        start_question_time = time.time()
        question = question.strip()
        if question[-1] != '?':
            question += '?'
            
        prompt = prepare_prompt(question, template_type=args.template_type)
        
        # Initialize the stopping criteria
        target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

        cnt = 0
        search_results_per_question = []
        search_results_ids = set()
        output_dict = {}

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

        end_question_time = time.time()
        total_inference_time += (end_question_time - start_question_time)
        
        
        # print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
        # print(prompt)
        # Encode the chat-formatted prompt and move it to the correct device
        while True:
            start_inference_time = time.time()
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            attention_mask = torch.ones_like(input_ids)
            
            # Generate text with the stopping criteria
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

            # if outputs[0][-1].item() in curr_eos:
            #     generated_tokens = outputs[0][input_ids.shape[1]:]
            #     output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            #     print(output_text)
            #     break

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            end_inference_time = time.time()
            total_inference_time += (end_inference_time - start_inference_time)
            
            start_retrieval_time = time.time()
            tmp_query, tmp_topk = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
            print('--------------------------------')
            print(f"tmp_query: {tmp_query}, tmp_topk: {tmp_topk}")
            print('--------------------------------')
            # if eos token is reached, break.
            # But before break, we need to add the last query to the output. Also collect the search results.
            if outputs[0][-1].item() in curr_eos:
                output_dict["last_query"] = tmp_query
                break

            if tmp_query:
                # print(f'searching "{tmp_query}"...')
                top_k = tmp_topk if tmp_topk is not None else args.topk
                search_results = search(tmp_query, top_k, args.port)
                # for search_result in search_results:
                    # print("--------------------------------")
                    # print(search_result)
                search_results_text, search_results_dicts = _passages2string(search_results)
                
                for search_result_dict in search_results_dicts:
                    if search_result_dict['id'] not in search_results_ids:
                        search_results_ids.add(search_result_dict['id'])
                        search_results_per_question.append(search_result_dict)
            else:
                search_results_text = ''
            end_retrieval_time = time.time()
            total_retrieval_time += (end_retrieval_time - start_retrieval_time)
            
            search_text = curr_search_template.format(output_text=output_text, search_results=search_results_text)
            prompt += search_text
            cnt += 1
            if cnt > args.max_count:
                break
            # print(search_text)

            
        output_dict["trajectory"] = prompt + "\n\n" + output_text
        output_dict["search_results"] = search_results_per_question
        output_dict["query_id"] = q_idx
        output_dict["total_retrieval_time"] = total_retrieval_time
        output_dict["total_inference_time"] = total_inference_time
        append_to_jsonl(args.output_file, output_dict)

        # Print full trajectory with <search> ... </search> and <information> ... </information> tags
        # print('\n\n################# [Full Trajectory] ##################\n')
        # print(output_dict["trajectory"])
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--model_id", type=str, default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo")
    parser.add_argument("--output_file", "-o", type=str, default="output.jsonl")
    parser.add_argument("--template_type", type=str, default="base", choices=["base", "dynamic"])
    parser.add_argument("--data_path", type=str, default="/scratch/hc3337/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_count", type=int, default=10)
    args = parser.parse_args()
    main(args)
