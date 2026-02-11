import os

# Project structure
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TOPIC_MODEL_DIR = os.path.join(PROJECT_ROOT, "topic_model_data")

# Kaleido topics configuration
NUM_KALEIDO_TOPICS = 40
"""
1 of the NUM_KALEIDO_TOPICS is an outlier/noise topic.
Kaleido_sys also provides *valence* so we have a positive and negative version of each topic.
Therefore, we have (NUM_KALEIDO_TOPICS - 1) * 2 *valenced* value categories.
"""
NUM_VALENCED_VALUE_CATEGORIES = (NUM_KALEIDO_TOPICS - 1) * 2

# Data paths
PROMPTS_CSV = os.path.join(DATA_DIR, "prompts.csv")
GRADED_PROMPTS_AIRBENCH = os.path.join(DATA_DIR, "graded_prompts_airbench.jsonl")
GRADED_PROMPTS_HARMBENCH = os.path.join(DATA_DIR, "graded_prompts_harmbench.jsonl")
GRADED_PROMPTS_PATHS = [GRADED_PROMPTS_AIRBENCH, GRADED_PROMPTS_HARMBENCH]

# Kaleido output paths
KALEIDO_DIR = os.path.join(DATA_DIR, "kaleido")
TOP_K_KALEIDO_TOPICS_JSONL = os.path.join(KALEIDO_DIR, "graded_prompts_airbench_deepseek-chat_kaleido_top_k_topics.jsonl")
TOP_K_KALEIDO_TOPICS_NPY = os.path.join(KALEIDO_DIR, "graded_prompts_airbench_deepseek-chat_kaleido_top_k_topics.npy")

# Topic model paths
TOPIC_ID_TO_NAME_JSON = os.path.join(TOPIC_MODEL_DIR, f"topic_id_to_name_{NUM_KALEIDO_TOPICS}.json")

KALEIDO_MODEL = "allenai/kaleido-base"
TOPIC_LABEL_MODEL = "gpt-4o-2024-08-06"

PROLIFIC_PID_KEY = "PROLIFIC_PID"
RANDOM_SEED = 42

LIWC_CATEGORIES = ["adverb", "negate", "adj", "allnone", "cause", "tentat", "certitude", "tone_pos", "tone_neg", "polite", "moral", "need", "want", "lack", "curiosity"]