# Project 5: Music Genre Classification with Transfer Learning

## Objective
Classify music genres using a pre-trained CNN (VGG16) and spectrograms from the GTZAN dataset.

## Skills
- Transfer learning
- Audio processing (spectrograms)
- Project structuring
- PyTorch implementation

## Steps
1. Load and preprocess GTZAN audio files into spectrograms.
2. Use a pre-trained VGG16 model with custom layers.
3. Train with a structured workflow (freeze/unfreeze).
4. Evaluate on a test set.

## Setup
### Option 1: Miniconda
- Create and activate the environment:
  ```bash
  conda env create -f environment.yml
  conda activate deeplearningproject5
- Open notebook.ipynb in Cursor and run cells.
### Option 2: Pip
- Install dependencies:
  ```bash
  pip install -r requirements.txt
- Open notebook.ipynb in Cursor or Jupyter Notebook.

## Notes
- Dataset from Kaggle (https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification).
- Extract `genres_original.zip` from `archive.zip` to `./data/genres_original/` (not tracked by Git unless added).
- Expected path: `./data/genres_original/` should contain genre folders (e.g., `blues/`, `classical/`).