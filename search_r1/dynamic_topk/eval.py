import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from search_r1.search.retrieval_manager import RetrievalManager
from typing import Optional, Tuple, List, Dict
import re
from vllm import LLM, SamplingParams
import argparse
import os
from pathlib import Path

class VLLMInferenceEngine:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 16384,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = 42,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            seed=seed,
        )

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [item.outputs[0].text for item in outputs]


def run_vllm_inference(engine: VLLMInferenceEngine, evaluation_data: List[Dict]) -> List[str]:
    system_prompt = "You are an expert in document retrieval. You will be given a passage and a question. You need to determine if the passage is relevant and useful for answering the question."
    user_prompt = "Question:\n{question}\n\nPassage:\n{passage}\n\nIs this passage useful for answering the question?\nAnswer only \"Yes\" or \"No\"."

    prompts = [engine.build_prompt(system_prompt, user_prompt.format(question=item['question'], passage=item['passage'])) for item in evaluation_data]
    outputs = engine.generate(prompts)

    for idx, text in enumerate(outputs, start=1):
        print(f"=== Output {idx} ===")
        print(text.strip())

    return outputs




def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def collect_subqueries(data):
    subqueries = []
    num_queries_per_inst = []
    topks = []
    for item in data:
        trajectory = item['trajectory']
        cur_subqueries, cur_topks = _parse_search_tag(trajectory)
        subqueries.extend(cur_subqueries)
        topks.extend(cur_topks)
        num_queries_per_inst.append(len(cur_subqueries))
    return subqueries, topks, num_queries_per_inst


def _parse_search_tag(text: str) -> Tuple[Optional[str], Optional[int]]:
    pattern = re.compile(r"<search(?:\s+topk=(\d+))?\s*>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        print('No matches found')
        return [], []

    subqueries = []
    topks = []
    print('-----')
    for _match in matches:
        print(_match)
        topk_str, query = _match
        topk = int(topk_str) if topk_str else None
        query = query.strip()
        if query == 'query':
            continue
        subqueries.append(query)
        topks.append(topk)
    return subqueries, topks


def perform_retrieval(subqueries, retrieval_manager, retrieval_batch_size):
    all_search_results = []
    for i in range(0, len(subqueries), retrieval_batch_size):
        batch_subqueries = subqueries[i:i+retrieval_batch_size]
        search_results = retrieval_manager.batch_search(batch_subqueries)
        all_search_results.extend(search_results)
    return all_search_results



def load_model(model_path):
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    return model, tokenizer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--exp_data_path", type=str, default="verl_checkpoints/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo_inference_musique/")
    parser.add_argument("--eval_file_path", type=str, default="test_outputs.jsonl")
    parser.add_argument("--output_file", type=str, default="vllm_outputs.jsonl")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
        
    
    if (Path(args.exp_data_path) / 'test_outputs_with_search_results.jsonl').exists():
        print('[WARNING] Search results already exist, skipping retrieval and collecting subqueries')
        output_data = read_jsonl(os.path.join(args.exp_data_path, 'test_outputs_with_search_results.jsonl'))
    else:
        data = read_jsonl(os.path.join(args.exp_data_path, args.eval_file_path))
        TOPK = 50
        retrieval_batch_size = 512

        subqueries, topks, num_queries_per_inst = collect_subqueries(data)
        
        # # perform retrieval batch by batch
        print('===============================================')
        print('Start Performing Retrieval')
        print('===============================================')
        retrieval_manager = RetrievalManager(search_url=f'http://127.0.0.1:{args.port}/retrieve', topk=TOPK)
        all_search_results = perform_retrieval(subqueries, retrieval_manager, retrieval_batch_size)

        # create a JSONL file to store the search results 
        # each entry in a JSON, with the following fields:
        # - subquery: the subquery that was used for retrieval
        # - ctxs: the search results for the subquery
        # - predicted_topk: the topk that was predicted by the model
        output_data = []
        assert len(subqueries) == len(all_search_results), f'Length mismatch: {len(subqueries)} != {len(all_search_results)}'
        assert len(topks) == len(subqueries), f'Length mismatch: {len(topks)} != {len(subqueries)}'

        for subquery, topk, search_results in zip(subqueries, topks, all_search_results):
            assert len(search_results) == TOPK, f'Retrieval result length mismatch: {len(search_results)} != {TOPK}'
            output_data.append({
                'subquery': subquery,
                'ctxs': search_results,
                'predicted_topk': topk if topk is not None else 0
            })
        write_jsonl(os.path.join(args.exp_data_path, 'test_outputs_with_search_results.jsonl'), output_data)
        write_jsonl(os.path.join(args.exp_data_path, 'num_queries_per_inst.jsonl'), [{"num_queries": _num} for _num in num_queries_per_inst])
        print('===============================================')
        print('End Performing Retrieval')
        print('===============================================')

    if (Path(args.exp_data_path) / args.output_file).exists():
        print('[WARNING] VLLM outputs already exist, skipping VLLM inference')
        vllm_output_data = read_jsonl(os.path.join(args.exp_data_path, args.output_file))
    else:
        evaluation_data = []
        for item in output_data:
            for ctx in item['ctxs']:
                evaluation_data.append({
                    'question': item['subquery'],
                    'passage': ctx['text'],
                })
        print(f'Running VLLM Inference on {len(evaluation_data)} pairs of question and passage')
        engine = VLLMInferenceEngine(model_name=args.model_path, gpu_memory_utilization=args.gpu_memory_utilization)
        vllm_outputs = run_vllm_inference(engine, evaluation_data)

        # create a JSONL file to store the VLLM outputs
        # each entry in a JSON, with the following fields:
        # - question: the question that was used for retrieval
        # - passage: the passage that was used for retrieval
        # - vllm_output: the output of the VLLM model
        vllm_output_data = []
        for item, vllm_output in zip(evaluation_data, vllm_outputs):
            vllm_output_data.append({
                'question': item['question'],
                'passage': item['passage'],
                'vllm_output': vllm_output.strip()
            })
            
        write_jsonl(os.path.join(args.exp_data_path, args.output_file), vllm_output_data)
    
    ## Compute the first topk for every question that is flagged as "Yes" by the VLLM model
    prev_question = ""
    topk_for_subquery = []
    _rank = 0
    for item in vllm_output_data:
        if item['question'] != prev_question:  # skip question if it is the same as the previous one
            _rank += 1
            if item['vllm_output'].strip().lower() == 'yes':
                topk_for_subquery.append({
                    'subquery': item['question'],
                    'topk': _rank,
                })
                prev_question = item['question']
                _rank = 0
    
    write_jsonl(os.path.join(args.exp_data_path, 'topk_for_subquery.jsonl'), topk_for_subquery)
    
if __name__ == "__main__":
    main()