"""
This script supports linguistic analysis of prompts with LIWC

The LIWC-22 CLI is a commercial tool that must be installed separately. See https://www.liwc.app/
"""

from argparse import ArgumentParser
import os
import subprocess
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from src.constants import *

def parse_args():
  parser = ArgumentParser()
  parser.add_argument("--prompts_path", type=str, required=True)
  parser.add_argument("--annotations_path", type=str, required=True)
  parser.add_argument("--outputs_dir", type=str, required=True)
  return parser.parse_args()

def preprocess(args):
  # Load prompts and annotations
  prompts_df = pd.read_csv(args.prompts_path)
  annotations_df = pd.read_csv(args.annotations_path)

  prompts = []
  scores = []

  # Iterate through each prompt and calculate mean rating
  for idx, row in prompts_df.iterrows():
    prompt_idx = row['Question_Index']
    prompt_text = row['Question_Content']

    # Get ratings for this prompt
    rating_col = f'Rating_{prompt_idx}'
    if rating_col in annotations_df.columns:
      ratings = annotations_df[rating_col].dropna()
      mean_rating = ratings.mean()
      prompts.append(prompt_text)
      scores.append(mean_rating)

  return prompts, scores

if __name__ == "__main__":
  args = parse_args()

  # Create output directory if it doesn't exist
  os.makedirs(args.outputs_dir, exist_ok=True)

  prompts, scores = preprocess(args)

  for p, s in zip(prompts, scores):
    print(f"Prompt: {p}\nScore: {s}\n")

  prompts_df = pd.DataFrame(prompts, columns=['prompt'])

  # Define output paths using outputs_dir
  prompts_output_path = os.path.join(args.outputs_dir, "prompts.csv")
  liwc_output_path = os.path.join(args.outputs_dir, "prompts_analyzed.csv")
  results_output_path = os.path.join(args.outputs_dir, "result_summary.csv")

  prompts_df.to_csv(prompts_output_path, index=False)

  cmd_to_execute = ["LIWC-22-cli",
                    "--input", prompts_output_path,
                    "--output", liwc_output_path,
                    "--column-indices", "1",
                    "--mode", "wc",
                    "--include-categories", ",".join(LIWC_CATEGORIES)]
  subprocess.call(cmd_to_execute)

  out_df = pd.read_csv(liwc_output_path)

  out_df['score'] = scores

  # compute correlation between scores and each of cats
  corrs, pvals = [], []
  for cat in LIWC_CATEGORIES:
    corr, pval = pearsonr(out_df[cat], out_df['score'])
    corrs.append(corr)
    pvals.append(pval)
  pvals_corrected = multipletests(pvals, method='holm')[1]

  # create df from these results
  results_df = pd.DataFrame({
      "liwc_category": LIWC_CATEGORIES,
      "pearson_corr": corrs,
      "p": pvals,
      "p_holm": pvals_corrected
  })

  # case insensitive sort by absolute value of pearson_corr
  results_df = results_df.reindex(results_df['pearson_corr'].abs().sort_values(ascending=False).index)
  results_df.round(3).to_csv(results_output_path, index=False)