# PluriHarms

Public data and code for the PluriHarms benchmark: studying pluralistic human judgments on AI harm.

## Overview

PluriHarms is a benchmark dataset that captures diverse human perspectives on AI safety. Rather than assuming universal agreement on what constitutes harmful AI behavior, this dataset documents systematic variation in harm judgments across annotators with different backgrounds and values.

The dataset contains:
- **150 prompts** spanning a range of potentially harmful content types
- **100 human annotators** with diverse demographic backgrounds
- **~15,000 harm ratings** (100 annotators × 150 prompts; missing values indicate "Unsure" responses) on a 0-100 scale
- **Psychological factor scores** derived from questionnaire responses (MFQ, Schwartz values, IRI, etc.)
- **Prompt features** including harm categories, effect types, and value topics

## Paper

For full details on the methodology and findings, see our paper (accepted to ICLR 2026):

**PluriHarms: Benchmarking the Full Spectrum of Human Judgments on AI Harm**
[arXiv:2601.08951](https://arxiv.org/pdf/2601.08951)

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with all dependencies
uv pip install -e ".[dev]"
```

## Data Files

| File | Description |
|------|-------------|
| `data/prompts.csv` | 150 prompts with harm level and feature annotations (action categories, effect types, value topics) |
| `data/annotations.csv` | 100 participants with demographics, harm ratings (0-100), trial order, and psychological factor scores |
| `data/factor_loadings.csv` | Factor loadings for computing psychological factors from questionnaire responses |

See [`data/README.md`](data/README.md) for detailed column documentation.

## Data Tutorial

See [`data_tutorial.ipynb`](data_tutorial.ipynb) for a comprehensive guide to loading and exploring the data.

**Note:** We are currently cleaning up the analysis code from the paper and will release it soon.

## Quick Start

```python
import pandas as pd

# Load the data
prompts = pd.read_csv('data/prompts.csv')
annotations = pd.read_csv('data/annotations.csv')

# Get ratings for a specific prompt
prompt_idx = 1
rating_col = f'Rating_{prompt_idx}'
ratings = annotations[rating_col].dropna()

print(f"Prompt: {prompts[prompts['Question_Index'] == prompt_idx]['Question_Content'].values[0]}")
print(f"Mean rating: {ratings.mean():.1f}")
print(f"Std rating: {ratings.std():.1f}")
```

### Preparing Data for Analysis

The `data_loader` module provides functions to prepare the data for statistical analysis:

```python
from src import data_loader

# Prepare long-format data for regression/mixed-effects modeling
df_long = data_loader.prepare_data_for_analysis()

# Include z-scored demographic features
df_long = data_loader.prepare_data_for_analysis(include_demographics=True)
```

The resulting DataFrame contains one row per rating with:
- Z-scored rating and trial order
- Prompt features (harm level, action/effect/value categories)
- Participant features (psychological factor scores, optionally demographics)

### Encoding Demographics

To encode categorical demographics for regression:

```python
# Load and encode demographics
annotations = data_loader.load_annotations()
annotations = data_loader.encode_demographics(annotations)  # Adds *_num columns
annotations = data_loader.zscore_demographics(annotations)  # Adds *_zscore columns
```

## Citation

```bibtex
@inproceedings{li2026pluriharms,
  title={PluriHarms: Benchmarking the Full Spectrum of Human Judgments on AI Harm},
  author={Li, Jing-Jing and Mire, Joel and Fleisig, Eve and Pyatkin, Valentina and Collins, Anne and Sap, Maarten and Levine, Sydney},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
