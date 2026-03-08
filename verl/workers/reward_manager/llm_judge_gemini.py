from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import re
from tqdm import tqdm
from google import genai
from google.genai.types import GenerateContentConfig, ModelContent, UserContent
from .prompts import ANSWER_PROMPT, LLM_AS_A_JUDGE_PRMOPT
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
import requests
import concurrent.futures


################## helper functions, to be moved #####################

@retry(
    wait=wait_random_exponential(multiplier=2, max=60),
    stop=stop_after_attempt(10),
)
def call_gemini_api(client, model_name, prompt, max_output_tokens=512, temperature=0.2, top_p=0.95):
    """Call the Gemini API to generate content."""
    chat = client.chats.create(
        model=model_name,
        config=GenerateContentConfig(
            max_output_tokens=max_output_tokens,  # Adjust as needed
            temperature=temperature,  # Adjust as needed
            top_p=top_p,  # Adjust as needed
        ),
    )
    response = chat.send_message(prompt).text
    return response


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

class LLMJudgeRewardManagerGemini:
    """The reward manager.
    """

    def __init__(self, 
                 tokenizer, 
                 model_name='gemini-2.0-flash',
                 max_worker=16,
                 num_examine=0, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.model_name = model_name
        self.max_worker = max_worker
        self.client = genai.Client(vertexai=True, 
            project='zifengw-research',# os.environ.get('GOOGLE_CLOUD_PROJECT'),
            location='us-central1', #os.environ.get('GOOGLE_CLOUD_LOCATION')
            )

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
        
        # update reward tensor with validity
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            future_to_idx = {
                executor.submit(self.get_reward, input_for_reward): idx for idx, input_for_reward in enumerate(batch_input_for_reward)
            }
        
        result = []
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc="Computing rewards"):
            idx = future_to_idx[future]
            try:
                reward = future.result()
                result.append((idx, reward))
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                result.append((idx, 0.0))
        # sort the results by index
        result = [reward for (idx, reward) in sorted(result, key=lambda x: x[0])]

        reward_score = torch.tensor(result, dtype=torch.float32)
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


    def get_llm_as_a_judge_result(self, question, gold_answer, model_answer, temperature, max_output_tokens):
        """
        Call the Gemini API to judge the ansdwer.
        """
        llm_as_a_judge_prompt = LLM_AS_A_JUDGE_PRMOPT.format(
            question=question,
            model_answer=model_answer,
            gold_answer=gold_answer
        )

        llm_as_a_judge_response = call_gemini_api(client=self.client,
            model_name=self.model_name,
            prompt=llm_as_a_judge_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        judgement = extract_judgement(llm_as_a_judge_response)
        return llm_as_a_judge_response, judgement

    def get_reward(self, processed_data):
        response_str = processed_data['response_str']
        answer = extract_solution(response_str)
        # print(f"Model answer: {answer}")
        # print(f"Question: {processed_data['question']}",
        #       f"\nGold Answer: {processed_data['gold_answer']}")
        if answer is None:
            return 0.0
        else:
            response, judgement = self.get_llm_as_a_judge_result(
                question=processed_data['question'],
                gold_answer=processed_data['gold_answer'],
                model_answer=response_str,
                temperature=0.0,
                max_output_tokens=512
            )
            if judgement is None:
                print(f"Judgement is None for response: {response_str}")
                return 0.0
            elif judgement == 'yes':
                return 1.0
            elif judgement == 'no':
                return 0.0
            else:
                print(f"Unexpected judgement: {judgement} for response: {response_str}")
                return 0.0