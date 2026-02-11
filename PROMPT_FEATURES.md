# Kaleido Value Topic Features

This guide explains how to extract value features from prompts.

## Pipeline Overview

The feature extraction pipeline consists of three steps:

```
1. generate_kaleido.py          → Extract values/rights/duties from prompts
2. generate_kaleido_topics.py   → Cluster Kaleido outputs into topics
3. extract_top_valenced_kaleido_topics.ipynb → Extract top topics per prompt
```

## Step 1: Extract Kaleido Features

The first step uses Kaleido to extract values, rights, and duties from each prompt. 

```bash
python src/prompt_feature/kaleido/generate_kaleido.py
```

**What this does:**
- Loads prompts from the files specified in `src/constants.py` (`GRADED_PROMPTS_PATHS`)
- For each prompt, generates candidate values/rights/duties using Kaleido
- Saves results back to the same files with new `*_kaleido` columns

**Output:** Adds `*_kaleido` columns to your prompt files containing lists of value/right/duty dictionaries.

## Step 2: Cluster Kaleido Outputs into Topics

The second step uses BERTopic to cluster similar Kaleido outputs into coherent topics.

**Download the pre-trained topic model:** The trained topic model is too large for GitHub. [Download it](https://drive.google.com/file/d/1mkLdq-9-JJOKoOW39vMqWmNZeR6xX8DC/view?usp=sharing) and place it in the `topic_model_data/` directory.

Alternatively, set `force_rebuild=True` in `generate_kaleido_topics.py` to train the topic model from scratch.
```bash
python src/prompt_feature/kaleido_topic/generate_kaleido_topics.py
```

**What this does:**
- Extracts all Kaleido values/rights/duties from Step 1
- Uses BERTopic with HDBSCAN clustering to group similar themes
- Generates human-readable topic labels using GPT-4o
- Saves topic assignments and metadata

**Outputs:**
- `topic_model_data/ktm_40_model` - Trained BERTopic model (download separately or train)
- `topic_model_data/topic_info_40.csv` - Topic statistics
- `topic_model_data/topic_id_to_name_40.json` - Topic names
- `topic_model_data/summaries_docs_40.json` - Topic summaries with example documents

## Step 3: Extract Top Topics per Prompt with extract_top_valenced_kaleido_topics.ipynb

The third step extracts the top k most relevant topics for each prompt, which are used for downstream analysis. 

**What this does:**
- For each prompt, identifies the top K most relevant topics
- Separates topics by valence (supporting vs opposing the value/right/duty)
- Saves extracted features for downstream analysis

**Outputs:**
- `data/kaleido/graded_prompts_airbench_deepseek-chat_kaleido_top_k_topics.jsonl` - Top topics per prompt (JSON)
- `data/kaleido/graded_prompts_airbench_deepseek-chat_kaleido_top_k_topics.npy` - Topic feature matrix (NumPy)

## Troubleshooting

**Import errors:**
Make sure your environment is set up. See README.md for details.