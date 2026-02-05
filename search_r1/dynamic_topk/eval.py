import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from search_r1.search.retrieval_manager import RetrievalManager
from typing import Optional, Tuple, List
import re
from vllm import LLM, SamplingParams


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


def run_vllm_inference() -> None:
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    system_prompt = "You are a concise assistant. Answer in 2-3 sentences."
    user_prompt = "Explain what retrieval-augmented generation is."

    engine = VLLMInferenceEngine(model_name=model_name, gpu_memory_utilization=0.7)
    prompt = engine.build_prompt(system_prompt, user_prompt)
    outputs = engine.generate([prompt])

    for idx, text in enumerate(outputs, start=1):
        print(f"=== Output {idx} ===")
        print(text.strip())





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


data = read_jsonl('verl_checkpoints/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo_inference_musique_sm/test_outputs.jsonl')
port = 8000
TOPK = 50
retrieval_batch_size = 512

subqueries, topks, num_queries_per_inst = collect_subqueries(data)

# print('Start Printing Subqueries and Topks')
# print('subqueries:', subqueries)
# print('topks:', topks)
# print('num_queries_per_inst:', num_queries_per_inst)

# # perform retrieval batch by batch
print('===============================================')
print('Start Performing Retrieval')
print('===============================================')
retrieval_manager = RetrievalManager(search_url=f'http://127.0.0.1:{port}/retrieve', topk=TOPK)
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
write_jsonl('verl_checkpoints/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo_inference_musique_sm/test_outputs_with_search_results.jsonl', output_data)
write_jsonl('verl_checkpoints/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo_inference_musique_sm/num_queries_per_inst.jsonl', [{"num_queries": _num} for _num in num_queries_per_inst])


print('===============================================')
print('End Performing Retrieval')
print('===============================================')


run_vllm_inference()

