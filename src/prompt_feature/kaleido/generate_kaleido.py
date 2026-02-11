
import argparse
from src.prompt_feature.kaleido.kaleido_sys import KaleidoSys
from tqdm import tqdm
import pandas as pd
import os
from src.constants import *

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--kaleido_model", type=str, default=KALEIDO_MODEL)
  return parser.parse_args()

def compute_kaleido(prompt, kaleido_sys):
  """Compute kaleido features for a single prompt."""
  if isinstance(prompt, float) or prompt is None:
    print(f"Warning: prompt is not a string: {prompt}. Skipping kaleido computation.")
    return []

  result_df = kaleido_sys.get_candidates(prompt)
  result_dicts = result_df.to_dict(orient='records')

  # remove irrelevant 'action' key provided by KaleidoSys
  for result_dict in result_dicts:
    del result_dict['action']

  return result_dicts

def process_file(graded_prompts_path, kaleido_sys, prompt_cols):
  """
  Process a single graded prompts file, adding kaleido features for each prompt column.

  Load graded prompts dataframe, assumed to contain the following columns:
  'seed_prompt', 'seed_harm_level', 'level_0.0', 'level_0.1', ... , 'level_1.0'
  """
  df = pd.read_json(graded_prompts_path, lines=True)

  kaleido_results = {f'{prompt_col}_kaleido': [] for prompt_col in prompt_cols}
  for prompt_col in prompt_cols:
    kaleido_col = f"{prompt_col}_kaleido"
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {kaleido_col}"):
      prompt = row[prompt_col]
      result_dicts = compute_kaleido(prompt, kaleido_sys)
      kaleido_results[kaleido_col].append(result_dicts)

    df[kaleido_col] = kaleido_results[kaleido_col]

    df.to_json(graded_prompts_path, orient='records', lines=True)

def main():
  args = parse_args()

  kaleido_sys = KaleidoSys(model_name=args.kaleido_model, use_tqdm=False)

  prompt_cols = ['seed_prompt']
  prompt_cols.extend(f"level_0.{i}" for i in range(0, 10))
  prompt_cols.append("level_1.0")

  for graded_prompts_path in GRADED_PROMPTS_PATHS:
    process_file(graded_prompts_path, kaleido_sys, prompt_cols)
    
if __name__ == "__main__":
    main()