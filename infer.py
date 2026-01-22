import transformers
import torch
from tqdm import tqdm
import random
from datasets import load_dataset
import requests
import json
import argparse




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

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str, topk: int):
    payload = {
            "queries": [query],
            "topk": [topk],
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
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
    # Model ID and device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    # Initialize the tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    # Get Data
    raw_data = read_jsonl(args.data_path)
    questions = [item['org_q'] for item in raw_data]


    # Inference Loop
    for q_idx, question in enumerate(tqdm(questions)):
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

        # print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
        # print(prompt)


        # Encode the chat-formatted prompt and move it to the correct device
        while True:
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
            
            tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
            
            # if eos token is reached, break.
            # But before break, we need to add the last query to the output. Also collect the search results.
            if outputs[0][-1].item() in curr_eos:
                output_dict["last_query"] = tmp_query
                break

            if tmp_query:
                # print(f'searching "{tmp_query}"...')
                search_results = search(tmp_query, args.topk)
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

            search_text = curr_search_template.format(output_text=output_text, search_results=search_results_text)
            prompt += search_text
            cnt += 1
            # print(search_text)

            
        output_dict["trajectory"] = prompt + "\n\n" + output_text
        output_dict["search_results"] = search_results_per_question
        output_dict["query_id"] = q_idx
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
    args = parser.parse_args()
    main(args)
