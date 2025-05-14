#!/usr/bin/env python3

import os
import sys
import json
import time
import argparse
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any
import requests

# Try loading API libraries, with helpful error messages if missing
try:
    import google.generativeai as genai
except ImportError:
    sys.exit("google-generativeai is not installed. Run:\n    pip install google-generativeai")

# Add Groq API support
try:
    from groq import Groq
except ImportError:
    sys.exit("groq is not installed. Run:\n    pip install groq")

# Model configurations - replaced HuggingFace models with Groq models
MODELS = OrderedDict([
    ("gemini-2.0-flash-lite", {"api": "gemini", "rpm": 30}),
    ("llama3-70b-8192", {"api": "groq", "rpm": 30}),
    ("llama-3.1-8b-instant", {"api": "groq", "rpm": 30}),
    ("llama-3.3-70b-versatile", {"api": "groq", "rpm": 30}),
    ("gemma2-9b-it", {"api": "groq", "rpm": 30}),
    ("allam-2-7b", {"api": "groq", "rpm": 30})
])

def load_dataset(json_path: str, set_json_path: str) -> Tuple[List[Dict], Dict]:
    """Load the JSON dataset and supplementary content."""
    # Load questions - read as a single JSON array instead of line-by-line
    with open(json_path, 'r') as f:
        # Skip the first line if it contains the file path comment
        content = f.read()
        if content.strip().startswith('//'):
            # Remove the first line that contains the comment
            content = content[content.find('\n')+1:]
        questions = json.loads(content)
    
    # Load supplementary content
    with open(set_json_path, 'r') as f:
        content = f.read()
        # Skip the first line if it contains a comment
        if content.strip().startswith('//'):
            content = content[content.find('\n')+1:]
        content_set = json.loads(content)
    
    return questions, content_set

def format_prompt(question: Dict, content_set: Dict, use_history: bool, history: List = None) -> str:
    """Format the prompt for the LLM, including background if present."""
    prompt = ""
    
    # Add background content if available
    if question["background"] != "none" and question["background"] in content_set:
        prompt += f"Background information:\n{content_set[question['background']]}\n\n"
    
    # Add the question
    prompt += f"Question: {question['question']}\n\n"
    
    # Add choices
    prompt += "Options:\n"
    for key, value in question['choices'].items():
        prompt += f"{key}. {value}\n"
    
    # Make instructions MUCH clearer for single letter response
    prompt += "\nCRITICAL INSTRUCTION: You must respond with EXACTLY ONE LETTER - either A, B, C, or D. DO NOT include any explanation, reasoning, or additional text. Your entire response should be just a single letter corresponding to the best answer."
    
    # Add conversation history if using history
    if use_history and history:
        conversation = "Previous questions and answers:\n\n"
        for prev in history:
            conversation += f"Question: {prev['question']}\n"
            conversation += "Options:\n"
            for key, value in prev['choices'].items():
                conversation += f"{key}. {value}\n"
            conversation += f"Correct answer: {prev['answer']}\n\n"
        
        prompt = conversation + "\nNew question:\n" + prompt
    
    return prompt

def extract_answer(response: str) -> Optional[str]:
    """Extract the answer (A, B, C, or D) from the model's response."""
    # First, clean up the response
    response = response.strip().upper()
    
    # If the response is just a single letter, return it
    if response in ["A", "B", "C", "D"]:
        return response
    
    # Otherwise, look for the first occurrence of A, B, C, or D
    for option in ["A", "B", "C", "D"]:
        if option in response:
            return option
            
    return None

def evaluate_model(model_name: str, model_config: Dict, questions: List[Dict], 
                  content_set: Dict, use_history: bool = False) -> Dict:
    """Evaluate a model on the dataset."""
    api_type = model_config["api"]
    rpm = model_config.get("rpm", 10)
    delay = 60.0 / rpm if rpm else 1.0
    
    results = []
    history = [] if use_history else None
    correct_count = 0
    
    for idx, question in tqdm(enumerate(questions), total=len(questions), 
                             desc=f"Evaluating {model_name}"):
        # Print current question information
        print(f"\n\n--- Question {idx+1}/{len(questions)} (ID: {question['id']}) ---")
        # print(f"Question: {question['question']}")
        print(f"Expected answer: {question['answer']}")
        
        prompt = format_prompt(question, content_set, use_history, history)
        
        try:
            if api_type == "gemini":
                response = call_gemini_api(model_name, prompt)
            elif api_type == "groq":
                response = call_groq_api(model_name, prompt)
            else:
                raise ValueError(f"Unknown API type: {api_type}")
            
            # Print raw response from model
            print(f"\nModel raw response:\n{response}")
                
            answer = extract_answer(response)
            is_correct = answer == question["answer"]
            
            # Print extracted answer and correctness
            # print(f"\nExtracted answer: {answer}")
            print(f"Correct: {'✅ YES' if is_correct else '❌ NO'}")
            
            if is_correct:
                correct_count += 1
                
            result = {
                "id": question["id"],
                "source": question["source"],
                "background": question["background"],
                "question": question["question"][:50] + "..." if len(question["question"]) > 50 else question["question"],
                "model_answer": answer,
                "correct_answer": question["answer"],
                "is_correct": is_correct
            }
            
            results.append(result)
            
            # Add to history if enabled
            if use_history:
                history.append(question)
                # Keep history to a reasonable length to avoid context length issues
                if len(history) > 5:
                    history.pop(0)
                    
            # Sleep to respect rate limits
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error evaluating question {question['id']}: {str(e)}")
            results.append({
                "id": question["id"],
                "source": question.get("source", "unknown"),
                "background": question.get("background", "none"),
                "question": question.get("question", "")[:50] + "..." if question.get("question") and len(question["question"]) > 50 else question.get("question", ""),
                "error": str(e),
                "correct_answer": question.get("answer"),
                "is_correct": False
            })
    
    accuracy = correct_count / len(questions) if questions else 0
    
    # Group accuracy by source
    source_accuracy = {}
    background_accuracy = {}
    
    for result in results:
        source = result.get("source", "unknown")
        if source not in source_accuracy:
            source_accuracy[source] = {"correct": 0, "total": 0}
        source_accuracy[source]["total"] += 1
        if result.get("is_correct", False):
            source_accuracy[source]["correct"] += 1
            
        background = result.get("background", "none")
        if background not in background_accuracy:
            background_accuracy[background] = {"correct": 0, "total": 0}
        background_accuracy[background]["total"] += 1
        if result.get("is_correct", False):
            background_accuracy[background]["correct"] += 1
    
    # Calculate percentages
    for source in source_accuracy:
        source_accuracy[source]["accuracy"] = (
            source_accuracy[source]["correct"] / source_accuracy[source]["total"]
        )
        
    for background in background_accuracy:
        background_accuracy[background]["accuracy"] = (
            background_accuracy[background]["correct"] / background_accuracy[background]["total"]
        )
    
    return {
        "model": model_name,
        "use_history": use_history,
        "accuracy": accuracy,
        "source_accuracy": source_accuracy,
        "background_accuracy": background_accuracy,
        "results": results
    }

def call_gemini_api(model_name: str, prompt: str) -> str:
    """Call the Gemini API."""
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

def call_groq_api(model_name: str, prompt: str) -> str:
    """Call the Groq API."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=10  # Limit token generation to encourage brevity
    )
    return response.choices[0].message.content

def generate_results_table(all_results: List[Dict]) -> pd.DataFrame:
    """Generate a results table."""
    rows = []
    
    for result in all_results:
        model_name = result["model"]
        history = "with_history" if result["use_history"] else "no_history"
        accuracy = result["accuracy"]
        
        # Add overall accuracy
        rows.append({
            "Model": model_name,
            "History": history,
            "Category": "Overall",
            "Subcategory": "All",
            "Accuracy": accuracy
        })
        
        # Add source accuracies
        for source, data in result["source_accuracy"].items():
            rows.append({
                "Model": model_name,
                "History": history,
                "Category": "Source",
                "Subcategory": source,
                "Accuracy": data["accuracy"]
            })
            
        # Add background accuracies
        for background, data in result["background_accuracy"].items():
            rows.append({
                "Model": model_name,
                "History": history,
                "Category": "Background",
                "Subcategory": background,
                "Accuracy": data["accuracy"]
            })
    
    return pd.DataFrame(rows)

def plot_results(results_df: pd.DataFrame, output_dir: str):
    """Plot the results."""
    # Plot overall accuracy by model and history setting
    plt.figure(figsize=(12, 6))
    
    # Filter for overall accuracy
    overall_df = results_df[results_df["Category"] == "Overall"]
    
    # Pivot data for plotting
    pivot_df = overall_df.pivot(index="Model", columns="History", values="Accuracy")
    
    # Plot grouped bar chart
    ax = pivot_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Model Performance with and without History")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.savefig(f"{output_dir}/overall_accuracy.png", dpi=300)
    
    # Plot accuracy by source
    source_df = results_df[results_df["Category"] == "Source"]
    
    plt.figure(figsize=(14, 8))
    pivot_df = source_df.pivot_table(
        index=["Model", "History"], 
        columns="Subcategory", 
        values="Accuracy"
    )
    
    ax = pivot_df.plot(kind="bar", figsize=(14, 8))
    plt.title("Model Performance by Source")
    plt.xlabel("Model / History Setting")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/source_accuracy.png", dpi=300)

def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on HLE72 dataset")
    parser.add_argument("--json", default="hle72.json", help="Path to the JSON dataset")
    parser.add_argument("--set", default="set.json", help="Path to the supplementary content")
    parser.add_argument("--history", choices=["on", "off", "both"], default="both",
                       help="Whether to use conversation history")
    parser.add_argument("--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("❌ GOOGLE_API_KEY environment variable not set.")
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("❌ GROQ_API_KEY environment variable not set.")
    
    # Configure APIs
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load dataset
    questions, content_set = load_dataset(args.json, args.set)
    print(f"Loaded {len(questions)} questions and {len(content_set)} background content items")
    
    # Determine history settings to evaluate
    history_settings = []
    if args.history in ["on", "both"]:
        history_settings.append(True)
    if args.history in ["off", "both"]:
        history_settings.append(False)
    
    all_results = []
    
    # Evaluate each model
    for model_name, model_config in MODELS.items():
        for use_history in history_settings:
            history_str = "with history" if use_history else "without history"
            print(f"\n➤ Evaluating {model_name} {history_str}")
            
            result = evaluate_model(
                model_name, 
                model_config, 
                questions, 
                content_set, 
                use_history
            )
            
            all_results.append(result)
            
            # Save individual results
            history_label = "with_history" if use_history else "no_history"
            result_file = f"{args.output}/{model_name.replace('/', '_')}_{history_label}.json"
            with open(result_file, 'w') as f:
                # Convert result data to serializable format
                serializable_result = {
                    "model": result["model"],
                    "use_history": result["use_history"],
                    "accuracy": result["accuracy"],
                    "source_accuracy": {k: v for k, v in result["source_accuracy"].items()},
                    "background_accuracy": {k: v for k, v in result["background_accuracy"].items()},
                    "results": result["results"]
                }
                json.dump(serializable_result, f, indent=2)
            
            print(f"    → Accuracy: {result['accuracy']:.2%}")
    
    # Generate and save results table
    results_df = generate_results_table(all_results)
    results_df.to_csv(f"{args.output}/results_summary.csv", index=False)
    
    # Format as a nice table for display
    print("\n📊 Results Summary")
    pivot_table = results_df[results_df["Category"] == "Overall"].pivot(
        index="Model", 
        columns="History", 
        values="Accuracy"
    )
    print(pivot_table)
    
    # Generate plots
    plot_results(results_df, args.output)
    print(f"\n✅ Evaluation complete. Results saved to {args.output} directory")

if __name__ == "__main__":
    main()