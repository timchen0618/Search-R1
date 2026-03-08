from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import re
from vllm import LLM
from vllm import SamplingParams
from .prompts import LLM_AS_A_JUDGE_PRMOPT

################## helper functions, to be moved #####################


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    if len(matches) == 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def extract_judgement(response):
    # extract "correct: yes\n\n" from the response
    correct_pattern = r'correct:\s*(yes|no)'
    match = re.search(correct_pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return None

###########################################################################

class LLMJudgeRewardManagerVLLM:
    """The reward manager.
    """

    def __init__(self, 
                 tokenizer, 
                 model_name='Qwen/Qwen3-32B',
                 num_examine=0, compute_score=None, tensor_parallel_size=4, gpu_memory_utilization=0.3) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.model_name = model_name
        self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)
        self.sampling_params = SamplingParams(max_tokens=512, temperature=0.0, top_p=0.95)

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # first decode the generated responses in batch
        prompt_ids = data.batch['prompts'] # batch_size x max_prompt_length
        response_ids = data.batch['responses'] # tensor
        bsz = prompt_ids.shape[0]
        prompt_length = prompt_ids.shape[1]
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=1)
        # decode
        batch_response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        batch_input_for_reward = [{'response_str': batch_response_str[i],
                                  'question': data.non_tensor_batch['question'][i],
                                  'gold_answer': data.non_tensor_batch['reward_model'][i]['ground_truth']['target']} for i in range(bsz)]

        # Batched vLLM inference: build prompts and run single generate call
        reward_score = self._compute_rewards_batched(batch_input_for_reward)
        print(f">> Computed rewards: {torch.mean(reward_score).item():.4f}")
        # if self.num_examine > 0:
        # print one randomly selected response and their reward
        idx = torch.randint(0, bsz, (1,)).item()
        print(f"Response: {batch_response_str[idx]}")
        print(f"Gold answer: {data.non_tensor_batch['reward_model'][idx]['ground_truth']['target']}")
        print(f"Reward: {reward_score[idx].item():.4f}")
        # update the reward tensor
        reward_tensor[torch.arange(bsz), valid_response_length - 1] = reward_score
        return reward_tensor


    def _compute_rewards_batched(self, batch_input_for_reward):
        """
        Compute rewards via batched vLLM inference. Builds all judge prompts,
        runs a single generate call, then extracts judgements.
        """
        bsz = len(batch_input_for_reward)
        reward_score = torch.zeros(bsz, dtype=torch.float32)

        # Build prompts and track which indices need judging
        prompts_to_judge = []
        indices_to_judge = []
        for idx, item in enumerate(batch_input_for_reward):
            answer = extract_solution(item['response_str'])
            if answer is None:
                reward_score[idx] = 0.0
            else:
                prompt = LLM_AS_A_JUDGE_PRMOPT.format(
                    question=item['question'],
                    model_answer=answer,
                    gold_answer=item['gold_answer']
                )
                prompts_to_judge.append(prompt)
                indices_to_judge.append(idx)

        if not prompts_to_judge:
            return reward_score

        # Single batched vLLM generate call
        outputs = self.vllm_engine.generate(
            prompts_to_judge,
            self.sampling_params,
            use_tqdm=False,
        )

        # Extract judgements and assign rewards
        for i, (idx, output) in enumerate(zip(indices_to_judge, outputs)):
            text = output.outputs[0].text
            judgement = extract_judgement(text)
            if judgement is None:
                print(f"Judgement is None for index {idx}")
                reward_score[idx] = 0.0
            elif judgement == 'yes':
                reward_score[idx] = 1.0
            elif judgement == 'no':
                reward_score[idx] = 0.0
            else:
                print(f"Unexpected judgement: {judgement} for index {idx}")
                reward_score[idx] = 0.0

        return reward_score