# Project 4: Sentiment Analysis on Movie Reviews

## Objective
Build an LSTM model in PyTorch to classify IMDB movie reviews as positive or negative.

## Skills
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM)
- Word embeddings
- Sequence data processing

## Steps
1. Load and preprocess the IMDB dataset.
2. Define an LSTM model with embeddings.
3. Train the model on review sequences.
4. Evaluate on a test set.

## Setup
### Option 1: Miniconda
- Create and activate the environment:
  ```bash
  conda env create -f environment.yml
  conda activate deeplearningproject4
- Open notebook.ipynb in Cursor and run cells.
### Option 2: Pip
- Install dependencies:
  ```bash
  pip install -r requirements.txt
- Open notebook.ipynb in Cursor or Jupyter Notebook.

## Notes
- Dataset downloaded from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz.
- Extracts to `./data/aclImdb/` (not tracked by Git unless added).