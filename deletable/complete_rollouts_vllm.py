import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from vllm import LLM, SamplingParams


def read_jsonl(file_path: str) -> List[Any]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(file_path: str, items: List[Dict[str, Any]]) -> None:
    with open(file_path, "a") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _extract_prompt(
    item: Any, input_key: str, fallback_keys: Tuple[str, ...]
) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if input_key in item and isinstance(item[input_key], str):
            return item[input_key]
        for key in fallback_keys:
            if key in item and isinstance(item[key], str):
                return item[key]
    return None


def _truncate_response(text: str) -> str:
    if "</search>" in text:
        return text.split("</search>")[0] + "</search>"
    if "</answer>" in text:
        return text.split("</answer>")[0] + "</answer>"
    return text


def _parse_action(text: str) -> Tuple[Optional[str], str, Optional[int]]:
    pattern = r"<(?P<action>search|answer)(?:\s+topk=(?P<topk>\d+))?>(?P<content>.*?)</(?P=action)>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None, "", None
    action = match.group("action")
    content = match.group("content").strip()
    topk = match.group("topk")
    return action, content, int(topk) if topk is not None else None


def _passages2string(retrieval_result: List[Dict[str, Any]]) -> str:
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return format_reference


def _batch_search(
    query_texts: List[str],
    topks: List[int],
    retrieval_url: str,
) -> List[str]:
    payload = {
        "queries": query_texts,
        "topk": topks,
        "return_scores": True,
    }
    results = requests.post(retrieval_url, json=payload).json()["result"]
    return [_passages2string(result) for result in results]


@dataclass
class RolloutState:
    input_index: int
    sample_id: int
    prompt: str
    raw_item: Any
    done: bool = False
    steps: int = 0
    terminated_reason: Optional[str] = None


def _batched(items: List[RolloutState], batch_size: int) -> List[List[RolloutState]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _run_generation_step(
    llm: LLM,
    states: List[RolloutState],
    sampling_params: SamplingParams,
    dynamic_topk: bool,
    default_topk: int,
    retrieval_url: str,
    allow_search: bool,
) -> None:
    prompts = [state.prompt for state in states]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    search_requests: List[Tuple[int, str, int]] = []
    parsed_actions: List[Tuple[RolloutState, str, str, Optional[int]]] = []

    for state, output in zip(states, outputs):
        response = output.outputs[0].text
        response = _truncate_response(response)

        action, content, topk = _parse_action(response)
        if action is None:
            parsed_actions.append((state, "invalid", response, None))
            continue
        if action == "search" and not allow_search:
            parsed_actions.append((state, "search_no_retrieval", response, topk))
            continue
        parsed_actions.append((state, action, response, topk))

        if action == "search" and allow_search:
            if dynamic_topk:
                query_topk = min(topk, 10) if topk is not None else default_topk
            else:
                query_topk = default_topk
            search_requests.append((len(search_requests), content, query_topk))

    search_results_text: List[str] = []
    if search_requests:
        query_texts = [content for _, content, _ in search_requests]
        topks = [topk for _, _, topk in search_requests]
        search_results_text = _batch_search(query_texts, topks, retrieval_url)

    search_idx = 0
    for state, action, response, _topk in parsed_actions:
        if action == "invalid":
            feedback = (
                "\nMy previous action is invalid. "
                "If I want to search, I should put the query between <search topk=N> and </search>, "
                "where N is an integer between 1 and 10, indicating the number of top results to return. "
                "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
                "Let me try again.\n"
                if dynamic_topk
                else "\nMy previous action is invalid. "
                "If I want to search, I should put the query between <search> and </search>. "
                "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
                "Let me try again.\n"
            )
            state.prompt += response + feedback
            state.steps += 1
            continue

        if action == "search":
            info_block = ""
            if allow_search and search_results_text:
                info_block = f"\n\n<information>{search_results_text[search_idx].strip()}</information>\n\n"
                search_idx += 1
            state.prompt += response + info_block
            state.steps += 1
            continue

        if action == "search_no_retrieval":
            state.prompt += response + "\n\n<information></information>\n\n"
            state.steps += 1
            continue

        if action == "answer":
            state.prompt += response
            state.done = True
            state.terminated_reason = "answer"
            state.steps += 1
            continue


def main(args: argparse.Namespace) -> None:
    raw_items = read_jsonl(args.input_file)

    base_entries: List[Tuple[int, str, Any]] = []
    for idx, item in enumerate(raw_items):
        prompt = _extract_prompt(item, args.input_key, ("trajectory", "prompt", "text"))
        if prompt is None:
            continue
        base_entries.append((idx, prompt, item))

    if not base_entries:
        raise ValueError("No valid prompts found in input JSONL.")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    llm = LLM(
        model=args.model_id,
        tokenizer=args.tokenizer or args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop,
        seed=args.seed,
    )

    all_states: List[RolloutState] = []
    for idx, prompt, item in base_entries:
        for sample_id in range(args.num_completions):
            all_states.append(
                RolloutState(
                    input_index=idx,
                    sample_id=sample_id,
                    prompt=prompt,
                    raw_item=item,
                )
            )

    for _ in range(args.max_turns):
        active_states = [state for state in all_states if not state.done]
        if not active_states:
            break
        for batch in _batched(active_states, args.batch_size):
            _run_generation_step(
                llm=llm,
                states=batch,
                sampling_params=sampling_params,
                dynamic_topk=args.dynamic_topk,
                default_topk=args.topk,
                retrieval_url=args.retrieval_url,
                allow_search=True,
            )

    active_states = [state for state in all_states if not state.done]
    if active_states:
        for batch in _batched(active_states, args.batch_size):
            _run_generation_step(
                llm=llm,
                states=batch,
                sampling_params=sampling_params,
                dynamic_topk=args.dynamic_topk,
                default_topk=args.topk,
                retrieval_url=args.retrieval_url,
                allow_search=False,
            )
        for state in active_states:
            if not state.done:
                state.done = True
                state.terminated_reason = "max_turns"

    base_prompt_map: Dict[int, str] = {}
    for idx, prompt, _item in base_entries:
        base_prompt_map[idx] = prompt

    grouped: Dict[int, Dict[str, Any]] = {}
    for state in all_states:
        base_prompt = base_prompt_map.get(state.input_index, "")
        entry = grouped.setdefault(
            state.input_index,
            {
                "input_index": state.input_index,
                "input": base_prompt,
                "raw_item": state.raw_item,
                "num_completions": 0,
                "completions": [],
            },
        )
        full_trajectory = state.prompt
        completion = (
            full_trajectory[len(base_prompt) :]
            if base_prompt and full_trajectory.startswith(base_prompt)
            else full_trajectory
        )
        entry["completions"].append(
            {
                "sample_id": state.sample_id,
                "completion": completion,
                "full_trajectory": full_trajectory,
                "steps": state.steps,
                "terminated_reason": state.terminated_reason,
            }
        )
        entry["num_completions"] += 1

    write_jsonl(args.output_file, list(grouped.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="outputs/completed_rollouts.jsonl")
    parser.add_argument("--input_key", type=str, default="trajectory")
    parser.add_argument("--num_completions", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--stop", type=str, nargs="+", default=["</search>", "</answer>"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--retrieval_url", type=str, default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--dynamic_topk", action="store_true")
    main(parser.parse_args())

