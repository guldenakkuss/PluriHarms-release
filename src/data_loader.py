"""
Data loading utilities for the PluriHarms dataset.

This module provides functions to load and prepare the PluriHarms dataset
for analysis. The data loading functions produce a DataFrame compatible
with the analysis code.

Note on Factor Scores
---------------------
The dataset contains Factor_1, Factor_2, and Factor_3 scores derived from
psychological questionnaire responses (MFQ, Schwartz values, IRI, Mini-IPIP,
AI literacy). These factors were computed using principal component analysis
on z-scored questionnaire means.

The three factors can be interpreted as:
- Factor_1: Captures power/achievement vs universalism/benevolence values
- Factor_2: Captures empathy and fairness-related moral foundations
- Factor_3: Captures tradition/conformity vs hedonism/stimulation values
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_annotations(data_path: str = "data/annotations.csv") -> pd.DataFrame:
    """
    Load the annotations data (participant demographics, ratings, and features).

    Args:
        data_path: Path to the annotations CSV file

    Returns:
        DataFrame with participant data including:
        - Participant_ID: Unique participant identifier
        - Demographics: Race_Ethnicity, Gender, Sexual_Orientation, Age,
          Political_Affiliation, Education, Religion_Importance,
          Social_Media_Frequency, Toxicity_Experience, Income
        - Ratings: Rating_1 through Rating_150 (harm ratings 0-100).
          Missing values (NaN) indicate the participant selected "Unsure".
        - Factor scores: Factor_1, Factor_2, Factor_3
        - Trial order: Trial_1 through Trial_150 (display order, raw integers)
    """
    return pd.read_csv(data_path)


def load_prompts(prompts_path: str = "data/prompts.csv") -> pd.DataFrame:
    """
    Load the prompts with their features.

    Args:
        prompts_path: Path to the prompts CSV file

    Returns:
        DataFrame with columns:
        - Question_Index: Integer index (1-150)
        - Question_Content: The prompt text
        - Harm_Level: Prompt harm level (raw, 0-1 scale)
        - action_*: 16 harm action category features (raw probabilities, sum to 1)
        - effect_*: 7 harm effect category features (raw probabilities, sum to 1)
        - value_topic_*: 39 value topic features (raw differences)

    Note: Features are stored as raw values. Use prepare_data_for_analysis()
    to get z-scored features suitable for regression.
    """
    return pd.read_csv(prompts_path)


def load_data(
    annotations_path: str = "data/annotations.csv",
    prompts_path: str = "data/prompts.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both annotations and prompts data.

    Args:
        annotations_path: Path to the annotations CSV file
        prompts_path: Path to the prompts CSV file

    Returns:
        Tuple of (annotations_df, prompts_df)
    """
    annotations_df = load_annotations(annotations_path)
    prompts_df = load_prompts(prompts_path)
    return annotations_df, prompts_df


# =============================================================================
# Column Helper Functions
# =============================================================================

def get_rating_columns(data_df: pd.DataFrame) -> List[str]:
    """Get the list of rating column names (Rating_1 through Rating_150)."""
    return [col for col in data_df.columns if col.startswith('Rating_')]


def get_trial_columns(data_df: pd.DataFrame) -> List[str]:
    """Get the list of trial column names (Trial_1 through Trial_150)."""
    return [col for col in data_df.columns if col.startswith('Trial_')]


def get_demographic_columns() -> List[str]:
    """Get the list of demographic column names."""
    return [
        'Race_Ethnicity', 'Gender', 'Sexual_Orientation', 'Age',
        'Political_Affiliation', 'Education', 'Religion_Importance',
        'Social_Media_Frequency', 'Toxicity_Experience', 'Income'
    ]


def get_factor_columns() -> List[str]:
    """Get the list of factor score column names."""
    return ['Factor_1', 'Factor_2', 'Factor_3']


def get_prompt_feature_columns(prompts_df: pd.DataFrame) -> dict:
    """
    Get lists of prompt feature columns by type.

    Returns:
        Dictionary with keys: 'action', 'effect', 'value'
    """
    return {
        'action': [c for c in prompts_df.columns if c.startswith('action_')],
        'effect': [c for c in prompts_df.columns if c.startswith('effect_')],
        'value': [c for c in prompts_df.columns if c.startswith('value_topic_')],
    }


def load_value_topic_names(path: str = "data/value_topic_names.json") -> dict:
    """
    Load the mapping from value topic numbers to human-readable names.

    Args:
        path: Path to the value_topic_names.json file

    Returns:
        Dictionary mapping topic number (as string) to topic name
    """
    import json
    with open(path) as f:
        return json.load(f)


def get_value_topic_name(topic_num: int, topic_names: dict = None) -> str:
    """
    Get the human-readable name for a value topic number.

    Args:
        topic_num: The topic number (0-38)
        topic_names: Optional pre-loaded topic names dict. If None, loads from file.

    Returns:
        The topic name (e.g., "Right to Privacy and Protection" for topic 0)

    Example:
        >>> get_value_topic_name(0)
        'Right to Privacy and Protection'
        >>> get_value_topic_name(1)
        'Freedom of Expression and Speech'
    """
    if topic_names is None:
        topic_names = load_value_topic_names()
    return topic_names.get(str(topic_num), f"Unknown topic {topic_num}")


# =============================================================================
# Demographic Encoding Functions
# =============================================================================

def encode_demographics(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical demographic variables to numeric values.

    This function converts categorical demographics to ordinal numeric values
    suitable for regression analysis. The encoding preserves meaningful ordinal
    relationships where they exist (e.g., age groups, education levels).

    Args:
        data_df: DataFrame with demographic columns

    Returns:
        DataFrame with additional numeric demographic columns (*_num)

    Encoding schemes:
        - Race_Ethnicity: White=1, Non-white=0
        - Gender: Man=1, Woman=0, Other=NaN
        - Sexual_Orientation: Straight=1, Other=0
        - Age: 18-24=0, 25-34=1, 35-44=2, 45-54=3, 55-64=4, 65+=5
        - Political_Affiliation: Very conservative=0 to Very liberal=4
        - Education: High school=0 to Graduate degree=4
        - Religion_Importance: Not at all=0 to Very important=3
        - Social_Media_Frequency: Less than weekly=0 to Over 2hrs/day=3
        - Toxicity_Experience: Rarely=0 to Very often=4
        - Income: <$20k=0 to >$100k=5
    """
    df = data_df.copy()

    # Race_Ethnicity: White = 1, Non-white = 0
    def encode_race(val):
        if pd.isna(val) or val in ['Unsure', 'Unanswered']:
            return np.nan
        val_lower = str(val).lower()
        if 'white' in val_lower and not any(x in val_lower for x in
            ['black', 'asian', 'hispanic', 'latino', 'native', 'pacific', 'mixed', 'other']):
            return 1
        return 0

    # Gender: Man = 1, Woman = 0
    def encode_gender(val):
        if pd.isna(val) or val in ['Unsure', 'Unanswered']:
            return np.nan
        val_lower = str(val).lower()
        if 'man' in val_lower and 'woman' not in val_lower:
            return 1
        elif 'woman' in val_lower:
            return 0
        return np.nan

    # Sexual_Orientation: Straight = 1, Other = 0
    def encode_orientation(val):
        if pd.isna(val) or val in ['Unsure', 'Unanswered']:
            return np.nan
        val_lower = str(val).lower()
        if 'straight' in val_lower or 'heterosexual' in val_lower:
            return 1
        return 0

    # Age: ordinal encoding
    age_mapping = {
        '18-24 years old': 0, '25-34 years old': 1, '35-44 years old': 2,
        '45-54 years old': 3, '55-64 years old': 4, '65 years old or above': 5,
        '65+ years old': 5
    }

    # Political_Affiliation: conservative to liberal scale
    political_mapping = {
        'Very conservative': 0, 'Somewhat conservative': 1, 'Moderate': 2,
        'Somewhat liberal': 3, 'Very liberal': 4
    }

    # Education: ordinal encoding
    def encode_education(val):
        if pd.isna(val):
            return np.nan
        val_lower = str(val).lower()
        if 'high school' in val_lower or 'ged' in val_lower:
            return 0
        elif 'some college' in val_lower:
            return 1
        elif 'associate' in val_lower or 'technical' in val_lower:
            return 2
        elif 'bachelor' in val_lower:
            return 3
        elif 'graduate' in val_lower or 'professional' in val_lower:
            return 4
        return np.nan

    # Religion_Importance: ordinal encoding
    religion_mapping = {
        'Not at all important': 0, ' Not at all important': 0,
        'Not very important': 1,
        'Somewhat important': 2, ' Somewhat important': 2,
        'Very important': 3, ' Very important': 3,
        'Extremely important': 4
    }

    # Social_Media_Frequency: ordinal encoding
    social_mapping = {
        'Less than once a week': 0, 'At least once a week': 1,
        'At least once a day': 2, 'Over 2 hours per day': 3
    }

    # Toxicity_Experience: ordinal encoding
    toxicity_mapping = {
        'Not a problem': 0, 'Rarely a problem': 1, 'Sometimes a problem': 2,
        'Often a problem': 3, 'Very often a problem': 4
    }

    # Income: ordinal encoding
    income_mapping = {
        'Less than $20,000': 0, '$20,000 to $34,999': 1, '$35,000 to $49,999': 2,
        '$50,000 to $74,999': 3, '$75,000 to $99,999': 4, 'Over $100,000': 5
    }

    # Apply encodings
    if 'Race_Ethnicity' in df.columns:
        df['Race_Ethnicity_num'] = df['Race_Ethnicity'].apply(encode_race)
    if 'Gender' in df.columns:
        df['Gender_num'] = df['Gender'].apply(encode_gender)
    if 'Sexual_Orientation' in df.columns:
        df['Sexual_Orientation_num'] = df['Sexual_Orientation'].apply(encode_orientation)
    if 'Age' in df.columns:
        df['Age_num'] = df['Age'].map(age_mapping)
    if 'Political_Affiliation' in df.columns:
        df['Political_Affiliation_num'] = df['Political_Affiliation'].map(political_mapping)
    if 'Education' in df.columns:
        df['Education_num'] = df['Education'].apply(encode_education)
    if 'Religion_Importance' in df.columns:
        df['Religion_Importance_num'] = df['Religion_Importance'].str.strip().map(
            {k.strip(): v for k, v in religion_mapping.items()})
    if 'Social_Media_Frequency' in df.columns:
        df['Social_Media_Frequency_num'] = df['Social_Media_Frequency'].map(social_mapping)
    if 'Toxicity_Experience' in df.columns:
        df['Toxicity_Experience_num'] = df['Toxicity_Experience'].map(toxicity_mapping)
    if 'Income' in df.columns:
        df['Income_num'] = df['Income'].map(income_mapping)

    return df


def zscore_demographics(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score the numeric demographic columns.

    Args:
        data_df: DataFrame with *_num demographic columns (from encode_demographics)

    Returns:
        DataFrame with additional z-scored demographic columns (*_zscore)
    """
    df = data_df.copy()

    num_cols = [c for c in df.columns if c.endswith('_num')]

    for col in num_cols:
        zscore_col = col.replace('_num', '_zscore')
        valid_data = df[col].dropna()

        if len(valid_data) > 1:
            mean_val = valid_data.mean()
            std_val = valid_data.std()

            if std_val > 0:
                df[zscore_col] = (df[col] - mean_val) / std_val
            else:
                df[zscore_col] = 0.0
        else:
            df[zscore_col] = np.nan

    return df


def get_demographic_num_columns() -> List[str]:
    """Get the list of numeric demographic column names."""
    return [
        'Race_Ethnicity_num', 'Gender_num', 'Sexual_Orientation_num', 'Age_num',
        'Political_Affiliation_num', 'Education_num', 'Religion_Importance_num',
        'Social_Media_Frequency_num', 'Toxicity_Experience_num', 'Income_num'
    ]


def get_demographic_zscore_columns() -> List[str]:
    """Get the list of z-scored demographic column names."""
    return [c.replace('_num', '_zscore') for c in get_demographic_num_columns()]


# =============================================================================
# Mixed Effects Data Preparation
# =============================================================================

def prepare_data_for_analysis(
    annotations_path: str = "data/annotations.csv",
    prompts_path: str = "data/prompts.csv",
    include_demographics: bool = False
) -> pd.DataFrame:
    """
    Prepare long-format data for mixed-effects modeling.

    This function creates a DataFrame suitable for mixed-effects regression
    by combining annotations.csv and prompts.csv.

    Args:
        annotations_path: Path to the annotations CSV file
        prompts_path: Path to the prompts CSV file
        include_demographics: If True, include z-scored demographic features

    Returns:
        DataFrame in long format with columns:
        - rating: Harm rating (z-scored)
        - participant_id: Participant identifier (for random effects)
        - question_number: Question identifier (for random effects)
        - trial: Display order (z-scored)
        - harm_level: Prompt harm level (z-scored)
        - prompt_text: The prompt text
        - action_*: Action category features (z-scored)
        - effect_*: Effect category features (z-scored)
        - value_topic_*: Value topic features (z-scored)
        - Factor_1, Factor_2, Factor_3: Psychological factor scores
        - *_zscore: Z-scored demographic features (if include_demographics=True)
    """
    print("Preparing data for mixed-effects modeling...")

    # Load data
    annotations_df = load_annotations(annotations_path)
    prompts_df = load_prompts(prompts_path)

    # Encode and z-score demographics if requested
    if include_demographics:
        print("  Encoding and z-scoring demographics...")
        annotations_df = encode_demographics(annotations_df)
        annotations_df = zscore_demographics(annotations_df)
        demographic_zscore_cols = [c for c in annotations_df.columns if c.endswith('_zscore')]
    else:
        demographic_zscore_cols = []

    # Get column lists
    rating_cols = get_rating_columns(annotations_df)
    trial_cols = get_trial_columns(annotations_df)
    factor_cols = get_factor_columns()

    prompt_feature_cols = (
        ['Harm_Level', 'Question_Content'] +
        get_prompt_feature_columns(prompts_df)['action'] +
        get_prompt_feature_columns(prompts_df)['effect'] +
        get_prompt_feature_columns(prompts_df)['value']
    )

    # Create long-format data
    long_data = []

    for _, participant_row in annotations_df.iterrows():
        participant_id = participant_row['Participant_ID']

        for i, rating_col in enumerate(rating_cols):
            question_number = int(rating_col.replace('Rating_', ''))
            rating = participant_row[rating_col]
            trial_col = f'Trial_{question_number}'

            if pd.notna(rating):
                # Get prompt features for this question
                prompt_row = prompts_df[prompts_df['Question_Index'] == question_number].iloc[0]

                row_data = {
                    'rating': float(rating) / 100.0,  # Convert to 0-1 scale
                    'participant_id': int(participant_id),
                    'question_number': question_number,
                    'trial': float(participant_row[trial_col]) if trial_col in participant_row else np.nan,
                    'harm_level': float(prompt_row['Harm_Level']),
                    'prompt_text': prompt_row['Question_Content'],
                }

                # Add prompt features (action, effect, value)
                for col in get_prompt_feature_columns(prompts_df)['action']:
                    row_data[col] = float(prompt_row[col])
                for col in get_prompt_feature_columns(prompts_df)['effect']:
                    row_data[col] = float(prompt_row[col])
                for col in get_prompt_feature_columns(prompts_df)['value']:
                    row_data[col] = float(prompt_row[col])

                # Add factor scores
                for col in factor_cols:
                    row_data[col] = float(participant_row[col])

                # Add demographic z-scores if requested
                for col in demographic_zscore_cols:
                    val = participant_row[col]
                    row_data[col] = float(val) if pd.notna(val) else np.nan

                long_data.append(row_data)

    df_long = pd.DataFrame(long_data)

    print(f"Created long-format dataset with {len(df_long)} observations")
    print(f"  Participants: {df_long['participant_id'].nunique()}")
    print(f"  Questions: {df_long['question_number'].nunique()}")

    # Z-score the ratings
    rating_mean = df_long['rating'].mean()
    rating_std = df_long['rating'].std()
    df_long['rating'] = (df_long['rating'] - rating_mean) / rating_std

    print(f"  Rating range after z-scoring: {df_long['rating'].min():.3f} to {df_long['rating'].max():.3f}")

    # Z-score the trial order
    trial_mean = df_long['trial'].mean()
    trial_std = df_long['trial'].std()
    df_long['trial'] = (df_long['trial'] - trial_mean) / trial_std

    # Z-score the prompt features (harm_level, action_*, effect_*, value_topic_*)
    # These are stored as raw values in prompts.csv and need to be z-scored for regression
    prompt_feature_cols_to_zscore = (
        ['harm_level'] +
        get_prompt_feature_columns(prompts_df)['action'] +
        get_prompt_feature_columns(prompts_df)['effect'] +
        get_prompt_feature_columns(prompts_df)['value']
    )

    for col in prompt_feature_cols_to_zscore:
        if col in df_long.columns:
            col_mean = df_long[col].mean()
            col_std = df_long[col].std()
            if col_std > 0:
                df_long[col] = (df_long[col] - col_mean) / col_std

    # Reorder columns to match expected output
    first_cols = ['rating', 'participant_id', 'question_number', 'trial',
                  'harm_level', 'prompt_text']
    action_cols = get_prompt_feature_columns(prompts_df)['action']
    effect_cols = get_prompt_feature_columns(prompts_df)['effect']
    value_cols = get_prompt_feature_columns(prompts_df)['value']

    col_order = first_cols + action_cols + effect_cols + value_cols + factor_cols + demographic_zscore_cols
    df_long = df_long[col_order]

    return df_long


def get_feature_columns(df_long: pd.DataFrame) -> dict:
    """
    Get lists of feature columns from the long-format dataframe.

    Args:
        df_long: Long-format DataFrame from prepare_data_for_analysis()

    Returns:
        Dictionary with keys:
        - 'action': Action category columns
        - 'effect': Effect category columns
        - 'value': Value topic columns
        - 'psychological_factor': Psychological factor score columns
        - 'demographic': Z-scored demographic columns (if included)
        - 'interaction': Interaction term columns (if any)
    """
    columns = df_long.columns.tolist()
    demographic_zscore_cols = get_demographic_zscore_columns()

    return {
        'action': [c for c in columns if c.startswith('action_')],
        'effect': [c for c in columns if c.startswith('effect_')],
        'value': [c for c in columns if c.startswith('value_topic_')],
        'psychological_factor': [c for c in columns if c.startswith('Factor_')],
        'demographic': [c for c in columns if c in demographic_zscore_cols],
        'interaction': [c for c in columns if '_x_' in c],
    }


# =============================================================================
# Statistics Functions
# =============================================================================

def compute_question_statistics(
    data_df: pd.DataFrame,
    prompts_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute statistics for each question.

    Args:
        data_df: DataFrame with rating columns
        prompts_df: Optional DataFrame with prompt features

    Returns:
        DataFrame with statistics for each question
    """
    rating_cols = get_rating_columns(data_df)

    stats = []
    for col in rating_cols:
        question_index = int(col.replace('Rating_', ''))
        ratings = data_df[col].dropna()

        stat = {
            'Question_Index': question_index,
            'Mean': ratings.mean(),
            'Std': ratings.std(),
            'Min': ratings.min(),
            'Max': ratings.max(),
            'Median': ratings.median(),
            'N_responses': len(ratings)
        }
        stats.append(stat)

    stats_df = pd.DataFrame(stats)

    if prompts_df is not None:
        stats_df = stats_df.merge(
            prompts_df[['Question_Index', 'Question_Content', 'Harm_Level']],
            on='Question_Index',
            how='left'
        )

    return stats_df.sort_values('Question_Index').reset_index(drop=True)


def compute_participant_statistics(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each participant.

    Args:
        data_df: DataFrame with rating columns

    Returns:
        DataFrame with statistics for each participant
    """
    rating_cols = get_rating_columns(data_df)
    demographic_cols = [col for col in get_demographic_columns() if col in data_df.columns]
    factor_cols = [col for col in get_factor_columns() if col in data_df.columns]

    ratings_only = data_df[rating_cols]

    stats_df = pd.DataFrame({
        'Participant_ID': data_df['Participant_ID'],
        'Mean_rating': ratings_only.mean(axis=1),
        'Std_rating': ratings_only.std(axis=1),
        'N_responses': ratings_only.notna().sum(axis=1)
    })

    for col in demographic_cols + factor_cols:
        stats_df[col] = data_df[col]

    return stats_df


# =============================================================================
# Convenience Functions
# =============================================================================

def get_question_content(question_index: int, prompts_df: pd.DataFrame) -> str:
    """Get the content/text of a specific question."""
    row = prompts_df[prompts_df['Question_Index'] == question_index]
    if len(row) == 0:
        raise ValueError(f"Question index {question_index} not found")
    return row['Question_Content'].iloc[0]


def get_ratings_for_question(
    question_index: int,
    data_df: pd.DataFrame,
    dropna: bool = True
) -> pd.Series:
    """
    Get all ratings for a specific question.

    Args:
        question_index: The question index (1-150)
        data_df: DataFrame with rating columns
        dropna: If True (default), exclude "Unsure" responses (NaN values)

    Returns:
        Series of ratings for the specified question
    """
    col = f'Rating_{question_index}'
    if col not in data_df.columns:
        raise ValueError(f"Column {col} not found in data")

    ratings = data_df[col]
    if dropna:
        ratings = ratings.dropna()
    return ratings
