#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# Replace proportions_ztest with a direct implementation
def proportions_ztest(count, nobs, alternative='two-sided'):
    """Simple implementation of two-proportion z-test."""
    p1 = count[0] / nobs[0]
    p2 = count[1] / nobs[1]
    p_pooled = (count[0] + count[1]) / (nobs[0] + nobs[1])
    
    z_stat = (p1 - p2) / np.sqrt(p_pooled * (1 - p_pooled) * (1/nobs[0] + 1/nobs[1]))
    
    # Calculate p-value (two-sided test)
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_val

from scipy import stats

def load_all_results(results_dir: str) -> Dict[str, Dict]:
    """Load all JSON result files from the results directory."""
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                model_name = data["model"]
                history_type = "with_history" if data["use_history"] else "no_history"
                key = f"{model_name}_{history_type}"
                results[key] = data
    return results

def calculate_history_gain(results: Dict[str, Dict]) -> pd.DataFrame:
    """Calculate the gain in accuracy when using history."""
    gain_data = []
    models = set()
    
    # Collect all unique models
    for key in results:
        model_name = key.split("_")[0]
        if model_name not in models:
            models.add(model_name)
    
    # Calculate gains for each model
    for model in models:
        with_hist_key = f"{model}_with_history"
        no_hist_key = f"{model}_no_history"
        
        if with_hist_key in results and no_hist_key in results:
            with_hist = results[with_hist_key]
            no_hist = results[no_hist_key]
            
            # Overall gain
            overall_gain = with_hist["accuracy"] - no_hist["accuracy"]
            gain_data.append({
                "Model": model,
                "Source": "Overall",
                "History Gain": overall_gain
            })
            
            # Source-specific gains
            for source in with_hist["source_accuracy"]:
                if source in no_hist["source_accuracy"]:
                    source_gain = with_hist["source_accuracy"][source]["accuracy"] - no_hist["source_accuracy"][source]["accuracy"]
                    gain_data.append({
                        "Model": model,
                        "Source": source,
                        "History Gain": source_gain
                    })
            
            # Background-specific gains
            for bg in with_hist["background_accuracy"]:
                if bg in no_hist["background_accuracy"]:
                    bg_gain = with_hist["background_accuracy"][bg]["accuracy"] - no_hist["background_accuracy"][bg]["accuracy"]
                    gain_data.append({
                        "Model": model,
                        "Background": bg,
                        "History Gain": bg_gain
                    })
    
    return pd.DataFrame(gain_data)

def calculate_block_consistency(results: Dict[str, Dict]) -> pd.DataFrame:
    """Calculate the block consistency for each model."""
    consistency_data = []
    
    for key, result in results.items():
        model_name = result["model"]
        history_type = "with_history" if result["use_history"] else "no_history"
        
        # Group questions by background
        bg_groups = defaultdict(list)
        for question in result["results"]:
            if question["background"] != "none":
                bg_groups[question["background"]].append(question)
        
        # Calculate consistency for each background
        consistent_blocks = 0
        for bg, questions in bg_groups.items():
            if all(q.get("is_correct", False) for q in questions):
                consistent_blocks += 1
        
        consistency = consistent_blocks / len(bg_groups) if bg_groups else 0
        
        consistency_data.append({
            "Model": model_name,
            "History": history_type,
            "Block Consistency": consistency,
            "Consistent Blocks": consistent_blocks,
            "Total Blocks": len(bg_groups)
        })
    
    return pd.DataFrame(consistency_data)

def calculate_refusal_rate(results: Dict[str, Dict]) -> pd.DataFrame:
    """Calculate the refusal rate (when model fails to provide a valid answer)."""
    refusal_data = []
    
    for key, result in results.items():
        model_name = result["model"]
        history_type = "with_history" if result["use_history"] else "no_history"
        
        # Count instances where model_answer is None
        refusals = 0
        total_questions = 0
        
        for question in result["results"]:
            total_questions += 1
            if question.get("model_answer") is None:
                refusals += 1
        
        refusal_rate = refusals / total_questions if total_questions > 0 else 0
        
        refusal_data.append({
            "Model": model_name,
            "History": history_type,
            "Refusal Rate": refusal_rate,
            "Refusals": refusals,
            "Total Questions": total_questions
        })
    
    return pd.DataFrame(refusal_data)

def calculate_ztest_significance(results: Dict[str, Dict]) -> pd.DataFrame:
    """Calculate z-test significance between with-history and no-history conditions."""
    ztest_data = []
    models = set()
    
    for key in results:
        model_name = key.split("_")[0]
        if model_name not in models:
            models.add(model_name)
    
    for model in models:
        with_hist_key = f"{model}_with_history"
        no_hist_key = f"{model}_no_history"
        
        if with_hist_key in results and no_hist_key in results:
            with_hist = results[with_hist_key]
            no_hist = results[no_hist_key]
            
            # Count successes and total questions
            with_hist_correct = sum(1 for q in with_hist["results"] if q.get("is_correct", False))
            with_hist_total = len(with_hist["results"])
            no_hist_correct = sum(1 for q in no_hist["results"] if q.get("is_correct", False))
            no_hist_total = len(no_hist["results"])
            
            # Perform z-test
            count = np.array([with_hist_correct, no_hist_correct])
            nobs = np.array([with_hist_total, no_hist_total])
            
            # Handle edge cases
            if with_hist_total > 0 and no_hist_total > 0:
                z_stat, p_val = proportions_ztest(count, nobs)
                
                ztest_data.append({
                    "Model": model,
                    "With History Correct": with_hist_correct,
                    "With History Total": with_hist_total,
                    "No History Correct": no_hist_correct,
                    "No History Total": no_hist_total,
                    "Z-statistic": z_stat,
                    "P-value": p_val,
                    "Significant": p_val < 0.05
                })
    
    return pd.DataFrame(ztest_data)

def calculate_position_decay(results: Dict[str, Dict]) -> pd.DataFrame:
    """Calculate how accuracy changes based on question position within a block."""
    position_data = []
    
    for key, result in results.items():
        model_name = result["model"]
        history_type = "with_history" if result["use_history"] else "no_history"
        
        # Group by background and maintain order
        bg_questions = defaultdict(list)
        for question in result["results"]:
            if question["background"] != "none":
                bg_questions[question["background"]].append(question)
        
        # Calculate position-based accuracy
        for bg, questions in bg_questions.items():
            sorted_questions = sorted(questions, key=lambda q: q["id"])  # Ensure order
            
            for i, question in enumerate(sorted_questions):
                position = i + 1  # 1-based position
                
                position_data.append({
                    "Model": model_name,
                    "History": history_type,
                    "Background": bg,
                    "Position": position,
                    "Correct": 1 if question.get("is_correct", False) else 0
                })
    
    return pd.DataFrame(position_data)

def calculate_passage_metrics(results: Dict[str, Dict]) -> pd.DataFrame:
    """Calculate long-passage loss and table coherence metrics."""
    metrics_data = []
    models = set()
    
    for key in results:
        model_name = key.split("_")[0]
        if model_name not in models:
            models.add(model_name)
    
    for model in models:
        with_hist_key = f"{model}_with_history"
        no_hist_key = f"{model}_no_history"
        
        if with_hist_key in results and no_hist_key in results:
            with_hist = results[with_hist_key]
            no_hist = results[no_hist_key]
            
            # Long-passage loss: NAEP - TIMSS
            naep_acc_w = with_hist["source_accuracy"].get("NAEP", {}).get("accuracy", 0)
            timss_acc_w = with_hist["source_accuracy"].get("TIMSS", {}).get("accuracy", 0)
            long_passage_loss_w = naep_acc_w - timss_acc_w
            
            naep_acc_n = no_hist["source_accuracy"].get("NAEP", {}).get("accuracy", 0)
            timss_acc_n = no_hist["source_accuracy"].get("TIMSS", {}).get("accuracy", 0)
            long_passage_loss_n = naep_acc_n - timss_acc_n
            
            # Table coherence: NASA with history - TIMSS with history
            nasa_acc_w = with_hist["source_accuracy"].get("NASA", {}).get("accuracy", 0)
            table_coherence = nasa_acc_w - timss_acc_w
            
            metrics_data.append({
                "Model": model,
                "Long Passage Loss (With History)": long_passage_loss_w,
                "Long Passage Loss (No History)": long_passage_loss_n,
                "Table Coherence": table_coherence
            })
    
    return pd.DataFrame(metrics_data)

def plot_history_gain(gain_df: pd.DataFrame, output_dir: str):
    """Plot history gain by model and source."""
    plt.figure(figsize=(12, 6))
    overall_gain = gain_df[gain_df["Source"] == "Overall"]
    
    # Sort by gain
    overall_gain = overall_gain.sort_values("History Gain", ascending=False)
    
    ax = sns.barplot(x="Model", y="History Gain", data=overall_gain, palette="viridis")
    plt.title("History Gain by Model")
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    
    # Add value labels
    for i, v in enumerate(overall_gain["History Gain"]):
        ax.text(i, v + 0.01 if v >= 0 else v - 0.03, 
                f"{v:.3f}", ha='center', fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, "history_gain.png"), dpi=300)

def plot_block_consistency(consistency_df: pd.DataFrame, output_dir: str):
    """Plot block consistency by model and history condition."""
    plt.figure(figsize=(12, 6))
    
    # Pivot for grouped bars
    pivot_df = consistency_df.pivot(index="Model", columns="History", values="Block Consistency")
    
    ax = pivot_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Block Consistency by Model")
    plt.ylabel("Block Consistency")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(title="")
    plt.tight_layout()
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.savefig(os.path.join(output_dir, "block_consistency.png"), dpi=300)

def plot_position_decay(position_df: pd.DataFrame, output_dir: str):
    """Plot accuracy by position in block."""
    plt.figure(figsize=(14, 8))
    
    # Group by model, history, and position to get average accuracy
    pos_acc = position_df.groupby(["Model", "History", "Position"])["Correct"].mean().reset_index()
    
    # Plot for each model
    models = pos_acc["Model"].unique()
    for i, model in enumerate(models):
        model_data = pos_acc[pos_acc["Model"] == model]
        
        plt.subplot(2, len(models)//2 + (1 if len(models) % 2 else 0), i+1)
        
        sns.lineplot(data=model_data, x="Position", y="Correct", hue="History", marker="o")
        plt.title(model)
        plt.ylabel("Accuracy" if i % (len(models)//2 + (1 if len(models) % 2 else 0)) == 0 else "")
        plt.ylim(0, 1.05)
        
    plt.suptitle("Position Decay Within Blocks", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "position_decay.png"), dpi=300)

def main():
    results_dir = "results"
    output_dir = "analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    results = load_all_results(results_dir)
    
    # Calculate metrics
    history_gain_df = calculate_history_gain(results)
    block_consistency_df = calculate_block_consistency(results)
    refusal_df = calculate_refusal_rate(results)
    ztest_df = calculate_ztest_significance(results)
    position_df = calculate_position_decay(results)
    passage_metrics_df = calculate_passage_metrics(results)
    
    # Save dataframes to CSV
    history_gain_df.to_csv(os.path.join(output_dir, "history_gain.csv"), index=False)
    block_consistency_df.to_csv(os.path.join(output_dir, "block_consistency.csv"), index=False)
    refusal_df.to_csv(os.path.join(output_dir, "refusal_rate.csv"), index=False)
    ztest_df.to_csv(os.path.join(output_dir, "ztest_significance.csv"), index=False)
    position_df.to_csv(os.path.join(output_dir, "position_decay.csv"), index=False)
    passage_metrics_df.to_csv(os.path.join(output_dir, "passage_metrics.csv"), index=False)
    
    # Create summary tables
    summary_df = pd.DataFrame({
        "Model": passage_metrics_df["Model"],
        "History Gain": history_gain_df[history_gain_df["Source"] == "Overall"]["History Gain"].values,
        "Block Consistency (with history)": block_consistency_df[block_consistency_df["History"] == "with_history"]["Block Consistency"].values,
        "Block Consistency (no history)": block_consistency_df[block_consistency_df["History"] == "no_history"]["Block Consistency"].values,
        "Refusal Rate (with history)": refusal_df[refusal_df["History"] == "with_history"]["Refusal Rate"].values,
        "Refusal Rate (no history)": refusal_df[refusal_df["History"] == "no_history"]["Refusal Rate"].values,
        "Z-test p-value": ztest_df["P-value"],
        "Significant Difference": ztest_df["Significant"],
        "Long Passage Loss (With History)": passage_metrics_df["Long Passage Loss (With History)"],
        "Table Coherence": passage_metrics_df["Table Coherence"]
    })
    
    summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    
    # Create plots
    plot_history_gain(history_gain_df, output_dir)
    plot_block_consistency(block_consistency_df, output_dir)
    plot_position_decay(position_df, output_dir)
    
    print(f"Analysis complete! Results saved to {output_dir} directory")
    
    # Display summary table
    print("\n====== SUMMARY OF METRICS ======")
    print(summary_df.to_string())
    print("\n======= STATISTICAL SIGNIFICANCE OF HISTORY =======")
    print(ztest_df[["Model", "Z-statistic", "P-value", "Significant"]].to_string(index=False))

if __name__ == "__main__":
    main()