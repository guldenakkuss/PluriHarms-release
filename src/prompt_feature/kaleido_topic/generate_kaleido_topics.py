from src.prompt_feature.kaleido_topic.topic_model import TopicModel
from src.prompt_feature.kaleido_topic.kaleido_document_classifier import KaleidoDocumentClassifier
import json
import os
from src.constants import *

# Create topic model.
topic_model = TopicModel(
    max_topics=NUM_KALEIDO_TOPICS,
    disambiguator=f"ktm_{NUM_KALEIDO_TOPICS}",
    ngram_range=(1, 5),    # N-gram range for vectorization
    min_topic_size=200,    # Minimum cluster size for HDBSCAN
    force_rebuild=False,    # Force model rebuild
    hdbscan_metric='euclidean', # Distance metric for HDBSCAN
    cluster_selection_method='eom'  # Cluster selection method
)

kaleido_cols = ['seed_prompt_kaleido']
kaleido_cols.extend(f"level_0.{i}_kaleido" for i in range(0, 10))
kaleido_cols.append("level_1.0_kaleido")

# Create document classifier (extracts documents at initialization)
classifier = KaleidoDocumentClassifier(GRADED_PROMPTS_PATHS, kaleido_cols, topic_model)

# Fit model (either new or cached) and classify
fitted_model, topics, probs = topic_model.fit(classifier.docs, classifier.doc_metadata)
classifier.classify_and_save(topics, probs)

# Get topic info and save to topic_model_data directory
bertopic_model = topic_model.topic_model
topic_info_path = os.path.join(TOPIC_MODEL_DIR, f"topic_info_{NUM_KALEIDO_TOPICS}.csv")
bertopic_model.get_topic_info().to_csv(topic_info_path, index=False)
print(f"Saved topic info to: {topic_info_path}")

# Get topic summaries and save to topic_model_data directory
topic_summaries, summaries_docs = topic_model.get_topic_summaries(classifier.docs)
print("Topic summaries:")
for summary, _docs in summaries_docs.items():
    print(f"{summary}: {len(_docs)}")

summaries_path = os.path.join(TOPIC_MODEL_DIR, f"summaries_docs_{NUM_KALEIDO_TOPICS}.json")
with open(summaries_path, "w", encoding="utf-8") as f:
    json.dump(summaries_docs, f, ensure_ascii=False, indent=2)
print(f"Saved topic summaries to: {summaries_path}")
