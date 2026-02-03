# PluriHarms Dataset

This directory contains the PluriHarms benchmark dataset for studying pluralistic human judgments on AI harm.

## Files

### `prompts.csv`
Contains the 150 prompts that annotators evaluated for potential harm, along with prompt-level features.

| Column | Description |
|--------|-------------|
| `Question_Index` | Integer index (1-150) identifying each prompt |
| `Question_Content` | The text of the prompt to be evaluated |
| `Harm_Level` | Prompt harm level (raw, 0-1 scale from classifier) |
| `action_*` | 16 harm action category features (raw probabilities from SafetyAnalyst, sum to 1) |
| `effect_*` | 7 harm effect category features (raw probabilities from SafetyAnalyst, sum to 1) |
| `value_topic_*` | 39 value topic features (raw differences from Kaleido topic model) |

**Note:** Prompt features are stored as raw values. Use `data_loader.prepare_data_for_analysis()` to get z-scored features suitable for regression.

### `annotations.csv`
Contains human annotations from 100 participants who rated all 150 prompts for perceived harm.

#### Demographic Columns

| Column | Description | Values |
|--------|-------------|--------|
| `Participant_ID` | Unique participant identifier | Integer (0-99) |
| `Race_Ethnicity` | Self-reported race/ethnicity | White, Black or African American, Latino or Hispanic, South Asian, Other, etc. |
| `Gender` | Self-reported gender | Man, Woman, Non-binary, etc. |
| `Sexual_Orientation` | Self-reported sexual orientation | Straight (heterosexual), Bisexual or pansexual, Lesbian, Gay, Aromantic or asexual, etc. |
| `Age` | Age range | 18-24 years old, 25-34 years old, 35-44 years old, 45-54 years old, 55-64 years old, 65+ years old |
| `Political_Affiliation` | Political leaning | Very liberal, Somewhat liberal, Moderate, Somewhat conservative, Very conservative, Other |
| `Education` | Highest education level | High school diploma or GED, Some college but no degree, Associates or technical degree, Bachelor's degree, Graduate or professional degree |
| `Religion_Importance` | Importance of religion in life | Very important, Somewhat important, Not at all important |
| `Social_Media_Frequency` | How often participant uses social media | Less than once a week, At least once a week, At least once a day, Over 2 hours per day |
| `Toxicity_Experience` | How often participant encounters toxic content online | Rarely a problem, Sometimes a problem, Often a problem, Very often a problem |
| `Income` | Household income range | Less than $20,000, $20,000 to $34,999, $35,000 to $49,999, $50,000 to $74,999, $75,000 to $99,999, Over $100,000 |

**Note:** For regression analysis, use `data_loader.encode_demographics()` and `data_loader.zscore_demographics()` to convert categorical demographics to z-scored numeric values.

#### Rating Columns

| Column | Description |
|--------|-------------|
| `Rating_1` to `Rating_150` | Harm ratings for each prompt (0-100 scale). The column `Rating_N` corresponds to the prompt with `Question_Index = N` in `prompts.csv`. Higher values indicate greater perceived harm. Missing values (NaN) indicate the participant selected "Unsure" for that prompt. |

#### Trial Order Columns

| Column | Description |
|--------|-------------|
| `Trial_1` to `Trial_150` | Display order for each prompt (raw integer values). The column `Trial_N` indicates when prompt N was shown to this participant during the annotation session. Lower values mean the prompt was shown earlier. |

#### Factor Scores

The factor scores are derived from psychological questionnaire responses using principal component analysis. Each factor is a weighted combination of z-scored questionnaire means.

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `Factor_1` | First principal component | Power/Achievement vs. Universalism/Benevolence values |
| `Factor_2` | Second principal component | Empathy, Fairness, and Perspective-taking |
| `Factor_3` | Third principal component | Tradition/Conformity vs. Hedonism/Stimulation values |

See `factor_loadings.csv` for the exact loadings used to compute these factors.

### `value_topic_names.json`
Maps value topic column indices to human-readable topic names. The topics are derived from the Kaleido topic model applied to ethical rights and duties.

Example usage:
```python
import json
with open('data/value_topic_names.json') as f:
    topic_names = json.load(f)
print(topic_names["0"])  # "Right to Privacy and Protection"
```

### `factor_loadings.csv`
Contains the factor loadings used to compute the three psychological factors from questionnaire responses. Provided for transparency and for researchers who wish to compute similar factors on their own data.

| Column | Description |
|--------|-------------|
| (index) | Z-scored questionnaire feature name (e.g., `MFQ_Harm_mean_z_scored`) |
| `Factor_1` | Loading weight for Factor 1 |
| `Factor_2` | Loading weight for Factor 2 |
| `Factor_3` | Loading weight for Factor 3 |

The questionnaire sources include:
- **MFQ**: Moral Foundations Questionnaire (Harm, Fairness, Ingroup, Authority, Purity)
- **Schwartz**: Schwartz Portrait Values (Power, Achievement, Hedonism, Stimulation, Self-direction, Universalism, Benevolence, Tradition, Conformity, Security)
- **IRI**: Interpersonal Reactivity Index (Empathic Concern, Fantasy, Personal Distress, Perspective Taking)
- **Mini-IPIP**: Openness to Experience scale
- **AI Literacy**: AI knowledge and attitudes scale

## Preparing Data for Analysis

Use `data_loader.prepare_data_for_analysis()` from `src/data_loader.py` to generate long-format data for mixed-effects modeling. This combines `annotations.csv` and `prompts.csv` into a single DataFrame with z-scored features.

## Rating Scale

Participants rated each prompt on a continuous scale from 0 to 100:
- **0** = Not harmful at all
- **100** = Extremely harmful
- **Unsure** = Participant was uncertain (stored as NaN in the data)

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{li2026pluriharms,
  title={PluriHarms: Benchmarking the Full Spectrum of Human Judgments on AI Harm},
  author={Li, Jing-Jing and Mire, Joel and Fleisig, Eve and Pyatkin, Valentina and Collins, Anne and Sap, Maarten and Levine, Sydney},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This dataset is released under the MIT License. See the LICENSE file in the root directory for details.
