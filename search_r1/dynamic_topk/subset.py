import json
from typing import Optional, Tuple, List, Dict
import re
import argparse
import os
from pathlib import Path


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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--exp_data_path", type=str, default="verl_checkpoints/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo_inference_musique/")
    parser.add_argument("--eval_file_path", type=str, default="test_outputs.jsonl")
    parser.add_argument("--output_file", type=str, default="vllm_outputs.jsonl")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
        
    TOPK = 50
    
    print('[WARNING] Search results already exist, skipping retrieval and collecting subqueries')
    search_results = read_jsonl(os.path.join(args.exp_data_path, 'test_outputs_with_search_results.jsonl'))
    
    # partition search results based on data subset.
    # HotpotQA(7405), 2WikiMultihopQA(12576), MuSiQue(2417), Bamboogle(125)
    search_results_partitioned = {'hotpotqa': [], '2wikimultihopqa': [], 'musique': [], 'bamboogle': []}
    subset_size = {'hotpotqa': 7405, '2wikimultihopqa': 12576, 'musique': 2417, 'bamboogle': 125}
    for idx, item in enumerate(search_results):
        if idx < 7405:
            search_results_partitioned['hotpotqa'].append(item)
        elif idx < 7405 + 12576:
            search_results_partitioned['2wikimultihopqa'].append(item)
        elif idx < 7405 + 12576 + 2417:
            search_results_partitioned['musique'].append(item)
        else:
            search_results_partitioned['bamboogle'].append(item)
            
    for subset, search_results in search_results_partitioned.items():
        print(f'Processing {subset} with {len(search_results)} search results')
        
    # create a subquery to dataset mapping
    subquery_to_dataset = {}
    for subset, search_results in search_results_partitioned.items():
        for item in search_results:
            subquery_to_dataset[item['subquery']] = subset

    ## Compute the first topk for every question that is flagged as "Yes" by the VLLM model
    print('[WARNING] Topk for subquery already exist, skipping computation')
    topk_for_subquery = read_jsonl(os.path.join(args.exp_data_path, 'topk_for_subquery.jsonl'))
    for item in topk_for_subquery:
        item['dataset'] = subquery_to_dataset[item['subquery']]

        
    ## Compute the statistics of the topks
    ## Also plot the distribution of the topks
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    # Extract the 'topk' values for the full dataset
    topks = [item['topk'] for item in topk_for_subquery]
    if len(topks) > 0:
        topks_array = np.array(topks)
        print("=== Aggregated Topk statistics ===")
        count = len(topks_array)
        print(f"  Count: {count}")
        print(f"  Min: {np.min(topks_array)}")
        print(f"  Max: {np.max(topks_array)}")
        print(f"  Mean: {np.mean(topks_array):.2f}")
        print(f"  Median: {np.median(topks_array)}")
        print(f"  Std: {np.std(topks_array):.2f}")
        print(f"  Quartiles: {np.percentile(topks_array, [25, 50, 75])}")

        # Compute average count per data (overall)
        total_data = sum(subset_size.values())
        avg_count_per_data = count / total_data if total_data > 0 else 0
        print(f"  Average count per data (all datasets): {avg_count_per_data:.5f}")

        # Plot histogram for full dataset
        # Bin topks into [1,2,3,...,10, '>10']
        def bin_topks(topks):
            binned = []
            for k in topks:
                if k > 10:
                    binned.append('>10')
                else:
                    binned.append(str(k))
            return binned

        bins_labels = [str(i) for i in range(1, 11)] + ['>10']

        binned_topks = bin_topks(topks)
        counts = Counter(binned_topks)
        total = sum(counts.values())
        percentages = [(counts.get(label, 0) / total * 100) if total > 0 else 0 for label in bins_labels]

        plt.figure(figsize=(8, 6))
        plt.bar(bins_labels, percentages, color="skyblue", edgecolor='black')
        plt.title("Distribution of First TopK where VLLM Output is 'Yes' (All Subsets)", fontsize=20)
        plt.xlabel("TopK", fontsize=16)
        plt.ylabel("Percentage (%)", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='y')
        plt.tight_layout()
        agg_hist_file = os.path.join(args.exp_data_path, "topk_distribution.png")
        plt.savefig(agg_hist_file)
        print(f"Histogram saved to {agg_hist_file}")
        plt.show()

        # Now per-subset statistics and plots
        subset_names = sorted(set(item['dataset'] for item in topk_for_subquery))
        for subset in subset_names:
            subset_topks = [item['topk'] for item in topk_for_subquery if item['dataset'] == subset]
            subset_topks_arr = np.array(subset_topks)
            print(f"\n=== Topk statistics for subset: {subset} ===")
            print(f"  Count: {len(subset_topks_arr)}")
            if len(subset_topks_arr) > 0:
                print(f"  Min: {np.min(subset_topks_arr)}")
                print(f"  Max: {np.max(subset_topks_arr)}")
                print(f"  Mean: {np.mean(subset_topks_arr):.2f}")
                print(f"  Median: {np.median(subset_topks_arr)}")
                print(f"  Std: {np.std(subset_topks_arr):.2f}")
                print(f"  Quartiles: {np.percentile(subset_topks_arr, [25, 50, 75])}")
                # Compute average count per data for this subset
                subset_total_data = subset_size.get(subset, 0)
                avg_count_per_data_subset = len(subset_topks_arr) / subset_total_data if subset_total_data > 0 else 0
                print(f"  Average count per data (for subset '{subset}'): {avg_count_per_data_subset:.5f}")

                binned_subset_topks = bin_topks(subset_topks)
                subset_counts = Counter(binned_subset_topks)
                subset_total = sum(subset_counts.values())
                subset_percentages = [(subset_counts.get(label, 0) / subset_total * 100) if subset_total > 0 else 0 for label in bins_labels]

                plt.figure(figsize=(8, 6))
                plt.bar(bins_labels, subset_percentages, color="lightgreen", edgecolor='black')
                plt.title(f"TopK Distribution: {subset}", fontsize=20)
                plt.xlabel("TopK", fontsize=16)
                plt.ylabel("Percentage (%)", fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(axis='y')
                plt.tight_layout()
                subset_hist_file = os.path.join(args.exp_data_path, f"topk_distribution_{subset}.png")
                plt.savefig(subset_hist_file)
                print(f"Histogram for subset '{subset}' saved to {subset_hist_file}")
                plt.show()
            else:
                print(f"No 'topk' values found for subset '{subset}'.")

    else:
        print("No 'topk' values found in topk_for_subquery.")
    
if __name__ == "__main__":
    main()