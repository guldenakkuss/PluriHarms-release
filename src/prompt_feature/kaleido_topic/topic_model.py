from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
import os
import pickle
import json
from collections import defaultdict
from tqdm import tqdm
from src.utils.llm_query_cache import LLMQueryCache
from hdbscan import HDBSCAN
from src.constants import *
import random
random.seed(RANDOM_SEED)

KALEIDO_SPECIFIC_STOP_WORDS = {'value', 'right', 'duty'}
ALL_STOP_WORDS = list(KALEIDO_SPECIFIC_STOP_WORDS.union(STOP_WORDS))
TOPIC_SUMMARY_PROMPT_TEMPLATE = """Below are prompts that have been clustered together as representing a similar topic. These prompts encode thematically-related values, rights, or duties.

Please analyze these documents and provide a concise topic name that captures the main value, right, or duty theme. The topic name should be:
- 3-6 words maximum
- Descriptive of the core ethical concept
- Focused on the shared value/right/duty theme

Avoid describing the valence/sentiment toward the value/right/duty theme. Output only the topic name, without any additional text or explanation.

Documents:
{docs_text}

Topic name:"""

class TopicModel:
    def __init__(self, 
                 max_topics,
                 disambiguator,
                 ngram_range=(1, 5),
                 min_topic_size=200,
                 force_rebuild=False,
                 hdbscan_metric='euclidean',
                 cluster_selection_method='eom'):
        self.max_topics = max_topics
        self.disambiguator = disambiguator
        self.ngram_range = ngram_range
        self.min_topic_size = min_topic_size
        self.hdbscan_metric = hdbscan_metric
        self.cluster_selection_method = cluster_selection_method
        self._force_rebuild = force_rebuild
        self.topic_model = None
        self.cached_docs = None
        self.cached_doc_metadata = None
        self.cached_topics = None
        self.cached_probs = None

        os.makedirs(TOPIC_MODEL_DIR, exist_ok=True)
        self.model_path = f"{TOPIC_MODEL_DIR}/{self.disambiguator}_model"
        self.cache_path = f"{TOPIC_MODEL_DIR}/{self.disambiguator}_cache.pkl"

        if force_rebuild:
            self.clear_model_and_cache()
        else:
            if os.path.exists(self.model_path):
                self.topic_model = BERTopic.load(path=self.model_path)
                self._load_cached_data()
    
    def clear_model_and_cache(self):
        print(f"Force rebuild enabled - clearing any existing model cache")
        # Remove the existing saved BERTopic model file
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            print(f"Removed old saved model: {self.model_path}")
        # Remove the cached fit artifacts (docs, metadata, topics, probs)
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            print(f"Removed cached topic model fit artifacts: {self.cache_path}")

    def fit(self, docs, doc_metadata):
        """Fit the topic model on the given documents and return topics and probabilities."""
        # Skip cache check if force_rebuild was requested during initialization
        if self.topic_model is not None and hasattr(self, '_force_rebuild') and not self._force_rebuild:
            # If model already exists, verify cached data matches input data
            if (self.cached_docs == docs and self.cached_doc_metadata == doc_metadata):
                return self.topic_model, self.cached_topics, self.cached_probs
            else:
                raise ValueError("Cached data does not match input data. Use force_rebuild=True to retrain.")
        elif self.topic_model is not None:
            print("Force rebuild: Ignoring cached model and retraining...")
        
        count_vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            stop_words=ALL_STOP_WORDS
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            metric=self.hdbscan_metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True  # Enable prediction data generation
        )
        
        self.topic_model = BERTopic(
            vectorizer_model=count_vectorizer,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=False,
            low_memory=True,  # Enable memory optimization for large datasets
            min_topic_size=self.min_topic_size,  # Prevent micro-clusters
            verbose=True
        )
        
        print("Starting BERTopic fit_transform...")
        try:
            topics, _ = self.topic_model.fit_transform(docs)
            print("BERTopic fit_transform completed successfully")
        except Exception as e:
            print(f"BERTopic fit_transform failed: {e}")
            raise RuntimeError(f"BERTopic fitting failed: {e}")
        
        # Reduce topics and get updated assignments
        print(f"Reducing from {len(set(topics))} to {self.max_topics} topics...")
        self.topic_model = self.topic_model.reduce_topics(
            docs=docs,
            nr_topics=self.max_topics
        )
        
        # Get updated topics and probabilities after reduction
        reduced_topics, reduced_probs = self.topic_model.transform(docs)
        print(f"Topics reduced to {len(set(reduced_topics))} final topics")

        self.save()
        self._save_cached_data(docs, doc_metadata, reduced_topics, reduced_probs)
    
        return self.topic_model, reduced_topics, reduced_probs
    
    def transform(self, docs):
        """Transform documents to get topics and probabilities."""
        if self.topic_model is None:
            raise ValueError("Model must be fitted before transforming documents")
        return self.topic_model.transform(docs)
    
    def get_topics_to_docs(self, docs):
        """Get mapping from topic IDs to documents."""
        topics, probs = self.transform(docs)
        
        topic_to_docs = defaultdict(list)
        for doc_idx, (topic, prob_dist) in enumerate(zip(topics, probs)):
            topic_to_docs[topic].append(docs[doc_idx])
        
        return dict(topic_to_docs)
    
    def get_topic_summaries(self, docs, model=TOPIC_LABEL_MODEL, force_rerun=False):
        """Generate summaries for each topic using LLM."""
        # Store LLM cache in topic_model_data directory
        os.makedirs(TOPIC_MODEL_DIR, exist_ok=True)
        cache_dir = os.path.join(TOPIC_MODEL_DIR, f"llm_cache_{NUM_KALEIDO_TOPICS}")
        llm_cache = LLMQueryCache(cache_dir=cache_dir, model=model)
        topic_to_docs = self.get_topics_to_docs(docs)
        topic_summaries = {}
        summaries_docs = {}
        
        valid_topics = [(tid, docs_list) for tid, docs_list in topic_to_docs.items() if tid != -1]
        
        random.seed(RANDOM_SEED)
        for topic_id, docs_list in tqdm(valid_topics, desc="Generating topic summaries"):
            sampled_docs = random.sample(docs_list, min(len(docs_list), 300))
            docs_text = "\n\n".join([f"{i+1}. {doc}" for i, doc in enumerate(sampled_docs)])

            prompt = TOPIC_SUMMARY_PROMPT_TEMPLATE.format(docs_text=docs_text)

            summary = llm_cache.query(prompt, force_rerun=force_rerun)
            topic_summaries[int(topic_id)] = summary.strip()
            summaries_docs[summary.strip()] = topic_to_docs[int(topic_id)]
        
        # Handle outlier topic
        if -1 in topic_to_docs:
            topic_summaries[-1] = "Outlier/Noise Topic"
            summaries_docs["Outlier/Noise Topic"] = topic_to_docs[-1]
        
        # Save topic ID to name mapping (convert keys to int and sort)
        topic_mapping_path = os.path.join(TOPIC_MODEL_DIR, f"topic_id_to_name_{NUM_KALEIDO_TOPICS}.json")
        
        # Convert string keys to int, sort by topic ID, then back to dict for JSON
        sorted_topic_summaries = dict(sorted(
            [(int(k), v) for k, v in topic_summaries.items()],
            key=lambda x: x[0]  # Sort by topic ID (int)
        ))
        
        with open(topic_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_topic_summaries, f, ensure_ascii=False, indent=2)
        print(f"Saved topic ID to name mapping to: {topic_mapping_path}")
        
        return topic_summaries, summaries_docs
    
    def save(self):
        """Save the BERTopic model to disk.

        Note: This only saves the model itself, not the cached fit artifacts
        (docs, metadata, topics, probs). Use _save_cached_data() for those.
        """
        if self.topic_model is None:
            raise ValueError("Model must be fitted before saving")
        self.topic_model.save(path=self.model_path)
    
    def load(self):
        """Load a BERTopic model from disk.

        Also loads the cached fit artifacts (docs, metadata, topics, probs)
        if they exist, for validation purposes.
        """
        self.topic_model = BERTopic.load(path=self.model_path)
        self._load_cached_data()
        return self.topic_model
    
    def _save_cached_data(self, docs, doc_metadata, topics, probs):
        """Save the inputs and outputs from fit() to a cache file.

        This caches the docs/metadata used during fit() along with the resulting
        topics/probs. Used to validate that subsequent fit() calls use the same
        data, preventing accidental retraining on different inputs.

        Saved separately from the model itself to {model_path}_cache.pkl
        """
        cache_data = {
            'docs': docs,
            'doc_metadata': doc_metadata,
            'topics': topics,
            'probs': probs
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_cached_data(self):
        """Load cached fit artifacts (docs, metadata, topics, probs).

        Loads the inputs/outputs from the original fit() call, stored separately
        from the model in {model_path}_cache.pkl. Used for validating that the
        same data is used in subsequent fit() calls.
        """
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.cached_docs = cache_data['docs']
                self.cached_doc_metadata = cache_data['doc_metadata']
                self.cached_topics = cache_data['topics']
                self.cached_probs = cache_data['probs']