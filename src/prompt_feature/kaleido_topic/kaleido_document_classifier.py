import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from collections import OrderedDict
import random
from src.constants import *

random.seed(RANDOM_SEED)

class KaleidoDocumentClassifier:
    def __init__(self, df_paths, kaleido_cols, topic_model):
        self.df_paths = df_paths
        self.kaleido_cols = kaleido_cols
        self.topic_model = topic_model
        self.docs, self.doc_metadata = self._extract_documents()
    
    def _extract_documents(self):
        """Extract documents and metadata from dataframes."""
        docs = []
        doc_metadata = []
        
        for df_path in self.df_paths:
            df = pd.read_json(df_path, lines=True)
            for i, row in df.iterrows():
                for kaleido_col in self.kaleido_cols:
                    kaleido_items = row[kaleido_col]
                    for item_idx, kaleido_item in enumerate(kaleido_items):
                        vrd_text = kaleido_item['text']
                        docs.append(vrd_text)
                        doc_metadata.append({
                            'df_path': df_path,
                            'row_index': i,
                            'kaleido_col': kaleido_col,
                            'item_index': item_idx
                        })
        
        return docs, doc_metadata
    
    def _add_risk_level_aggregations(self, df, num_topics=40, top_k=5):
        """Add risk-level aggregations using majority vote for topic assignments."""
        for kaleido_col in tqdm(self.kaleido_cols, desc="Risk-level aggregations", leave=False):
            topic_col_name = f'{kaleido_col}_topics'
            agg_topic_col_name = f'{kaleido_col}_agg_topic'
            df[agg_topic_col_name] = None

            top_k_valenced_topics_col_name = f'{kaleido_col}_top_k_valenced_topics'
            top_k_valenced_topics_valence_scores = f'{kaleido_col}_top_k_valenced_topics_valence_scores'
            df[top_k_valenced_topics_col_name] = None
            df[top_k_valenced_topics_valence_scores] = None

            for row_idx in df.index:
                topic_list = df.loc[row_idx, topic_col_name]
                kaleido_items = df.loc[row_idx, kaleido_col]
            
                """
                Add top k unique valenced topics
                """
                if topic_list and kaleido_items:
                    valenced = OrderedDict()
                    final_kaleido_items = []
                    for topic, kaleido_item in zip(topic_list, kaleido_items):
                        if topic == -1:  # skip outliers
                            continue
                        final_kaleido_items.append(kaleido_item)
                        is_supportive = kaleido_item['supports'] >= kaleido_item['opposes']
                        valenced_topic = topic if is_supportive else topic + num_topics - 1
                        valence_score = kaleido_item['supports'] if is_supportive else kaleido_item['opposes']
                        if valenced_topic not in valenced:
                            valenced[valenced_topic] = valence_score

                    final_valenced_topics = list(valenced.keys())[:top_k]
                    final_valence_scores = list(valenced.values())[:top_k]

                    df.at[row_idx, top_k_valenced_topics_col_name] = final_valenced_topics
                    df.at[row_idx, top_k_valenced_topics_valence_scores] = final_valence_scores
                else:
                    df.at[row_idx, top_k_valenced_topics_col_name] = []
                    df.at[row_idx, top_k_valenced_topics_valence_scores] = []          

    def _add_outlier_proportions(self, df):
        """Add proportion of outlier items for each kaleido column."""
        for kaleido_col in tqdm(self.kaleido_cols, desc="Calculating outlier proportions", leave=False):
            topic_col_name = f'{kaleido_col}_topics'
            outlier_prop_col_name = f'{kaleido_col}_outlier_proportion'
            
            # Initialize outlier proportion column
            df[outlier_prop_col_name] = None
            
            # Calculate for each row
            for row_idx in df.index:
                topic_list = df.loc[row_idx, topic_col_name]
                
                if topic_list and len(topic_list) > 0:
                    # Count outlier topics (topic == -1)
                    outlier_count = sum(1 for topic in topic_list if topic == -1)
                    total_count = len(topic_list)
                    
                    # Calculate proportion
                    outlier_proportion = outlier_count / total_count if total_count > 0 else 0.0
                    df.at[row_idx, outlier_prop_col_name] = round(outlier_proportion, 4)
                else:
                    df.at[row_idx, outlier_prop_col_name] = 0.0
    
    def classify_and_save(self, topics, probs):
        """Classify documents and save results to dataframes."""
        if topics is None:
            raise ValueError("Topics must be provided")
        if len(topics) != len(self.doc_metadata):
            print(f"Warning: Length mismatch! {len(topics)} topics vs {len(self.doc_metadata)} metadata entries. This will cause missing predictions.")
        
        # Group predictions by dataframe and row
        df_predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for topic, prob_dist, metadata in zip(topics, probs, self.doc_metadata):
            df_path = metadata['df_path']
            row_idx = metadata['row_index']
            kaleido_col = metadata['kaleido_col']
            item_idx = metadata['item_index']
            
            df_predictions[df_path][row_idx][kaleido_col][item_idx] = topic

        # Update each dataframe with topic predictions
        for df_path in tqdm(self.df_paths, desc="Processing dataframes"):
            df = pd.read_json(df_path, lines=True)
            
            # Add kaleido_topics column for each kaleido_col (skip membership scores when probabilities disabled)
            for kaleido_col in self.kaleido_cols:
                topic_col_name = f'{kaleido_col}_topics'
                df[topic_col_name] = None
                
                for row_idx in df.index:
                    kaleido_items = df.loc[row_idx, kaleido_col]
                    topic_predictions = []
                    
                    for item_idx in range(len(kaleido_items)):
                        # Get predictions with fallback for missing data
                        if item_idx in df_predictions[df_path][row_idx][kaleido_col]:
                            topic_predictions.append(df_predictions[df_path][row_idx][kaleido_col][item_idx])
                        else:
                            # Fallback for missing predictions
                            print(f"Warning: Missing prediction for {df_path} row {row_idx}, {kaleido_col}, item {item_idx}")
                            topic_predictions.append(-1)  # Assign as outlier
                    
                    df.at[row_idx, topic_col_name] = np.array(topic_predictions, dtype=np.int16).tolist()
            
            # Add risk-level aggregations for each kaleido_col
            self._add_risk_level_aggregations(df)
            
            # Add outlier proportion tracking for each kaleido_col
            self._add_outlier_proportions(df)
            
            # Save updated dataframe (overwrite original)
            print(f"Saving classified data to: {df_path}")
            df.to_json(df_path, orient='records', lines=True)