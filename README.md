# HLE72: Evaluating LLM Performance With and Without Conversation History

## Project Overview
HLE72 evaluates the impact of conversation history on large language models' performance across 72 educational questions spanning mathematics, science, and reading comprehension domains.

## Setup
1. Clone the repository
2. Create a `.env` file with API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

## Usage

### Evaluation
Run the evaluation script to test models:
```bash
# Test all models with and without history
python eval_hle72.py

# Only test with history
python eval_hle72.py --history on

# Only test without history
python eval_hle72.py --history off

# Customize output directory
python eval_hle72.py --output custom_dir
```

### Analysis
After running evaluation, analyze the results:
```bash
python analyze_results.py
```

## Output
- Evaluation (eval_hle72.py) generates result JSON files and a summary CSV in the results directory
- Analysis (analyze_results.py) generates metrics CSVs and visualizations in the analysis directory
