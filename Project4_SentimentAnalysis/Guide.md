Nice work on Project 3! A 70% accuracy on the Cats vs. Dogs classifier with a small subset is a solid start—CNNs often need more data or tuning to climb higher, but it’s great that it’s running smoothly. Now, let’s move to Project 4: **Sentiment Analysis on Movie Reviews**. This one shifts gears to sequence models, using an LSTM in PyTorch to classify IMDB movie reviews as positive or negative. We’ll set it up in your repo with the Miniconda + `pip` dual setup, and build it step-by-step in Cursor.

---

### Step 1: Setting Up Project 4 Folder

#### Folder Structure
Adding Project 4 to your repo:
```
DeepLearningPractice/
├── Project1_HandwrittenDigitClassifier/
│   ├── notebook.ipynb
│   ├── environment.yml
│   ├── requirements.txt
│   └── README.md
├── Project2_ImprovedDigitClassifier/
│   ├── notebook.ipynb
│   ├── environment.yml
│   ├── requirements.txt
│   └── README.md
├── Project3_CatsVsDogsClassifier/
│   ├── notebook.ipynb
│   ├── environment.yml
│   ├── requirements.txt
│   └── README.md
├── Project4_SentimentAnalysis/
│   ├── notebook.ipynb
│   ├── environment.yml
│   ├── requirements.txt
│   └── README.md
├── Project5_MusicGenreClassification/
│   └── README.md
├── README.md
└── .gitignore
```

#### Setup Instructions
1. **Navigate to Project 4 Folder**:
   - In Cursor, open `DeepLearningPractice/Project4_SentimentAnalysis/`.
   - `README.md` is already there from the initial setup.

2. **Add `environment.yml` (for Your Miniconda)**:
   - Create `environment.yml` in Cursor:
     ```yaml
     name: deeplearningproject4
     channels:
       - pytorch
       - conda-forge
       - defaults
     dependencies:
       - python=3.9
       - pytorch>=2.0.0
       - torchvision>=0.15.0
       - numpy>=1.24.0
       - matplotlib>=3.7.0
       - jupyter>=1.0.0
       - pip
     ```
   - Set up the environment:
     ```bash
     conda env create -f environment.yml
     conda activate deeplearningproject4
     ```
   - Note: Same dependencies as before; no extra NLP libraries needed since PyTorch handles the LSTM and embeddings.

3. **Add `requirements.txt` (for Pip Compatibility)**:
   - Create `requirements.txt` in Cursor:
     ```
     torch>=2.0.0
     torchvision>=0.15.0
     numpy>=1.24.0
     matplotlib>=3.7.0
     jupyter>=1.0.0
     ```

4. **Update `README.md`**:
   - Edit `README.md` in Cursor:
     ```markdown
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
       ```
     - Open `notebook.ipynb` in Cursor and run cells.

     ### Option 2: Pip
     - Install dependencies:
       ```bash
       pip install -r requirements.txt
       ```
     - Open `notebook.ipynb` in Cursor or Jupyter Notebook.

     ## Notes
     - Uses PyTorch’s built-in IMDB dataset (torchtext not required).
     - Data downloads to `./data/` (not tracked by Git unless added).
     ```

5. **Create `notebook.ipynb`**:
   - In Cursor, create `notebook.ipynb` in `Project4_SentimentAnalysis/`.

---

### Step 2: Implementing Project 4 in `notebook.ipynb`

Activate your Conda environment (`conda activate deeplearningproject4`), open `notebook.ipynb` in Cursor, and add these cells. We’ll use PyTorch’s IMDB dataset and build an LSTM for sentiment classification.

#### Cell 1: Imports
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
```
- **Run**: Imports for LSTM and data handling. We’ll use `torchvision.datasets` for IMDB (despite its image focus, it includes text datasets).

#### Cell 2: Load and Preprocess IMDB Dataset
```python
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load IMDB dataset
vocab_size = 10000  # Limit to 10K most frequent words
max_len = 200  # Max sequence length

# Define a simple tokenizer (split words, truncate/pad)
def tokenize(text):
    words = text.lower().split()
    return words[:max_len] + ['<pad>'] * (max_len - len(words)) if len(words) < max_len else words[:max_len]

# Load dataset with basic preprocessing
train_dataset = datasets.IMDB(root='./data', split='train')
test_dataset = datasets.IMDB(root='./data', split='test')

# Build vocabulary
word2idx = {'<pad>': 0, '<unk>': 1}
for text, _ in train_dataset:
    for word in tokenize(text):
        if word not in word2idx and len(word2idx) < vocab_size:
            word2idx[word] = len(word2idx)

# Convert text to indices
def text_to_tensor(text):
    indices = [word2idx.get(word, 1) for word in tokenize(text)]  # 1 is <unk>
    return torch.tensor(indices, dtype=torch.long)

train_data = [(text_to_tensor(text), label - 1) for text, label in train_dataset]  # Label 1→0, 2→1
test_data = [(text_to_tensor(text), label - 1) for text, label in test_dataset]

# Train/validation split
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Visualize a sample
sample_text, sample_label = train_dataset[0]
print(f'Sample Label: {"Positive" if sample_label == 1 else "Negative"}')
print(f'Sample Text (first 20 words): {[k for k, v in word2idx.items() if v in sample_text[:20]]}')
```
- **Setup**: 
  - Uses `torchvision.datasets.IMDB` (downloads to `./data/`).
  - Limits vocab to 10K words, pads/truncates to 200 words.
  - Simple tokenizer (splits on spaces).
- **Run**: Downloads IMDB, builds vocab, and shows a sample.

#### Cell 3: Define the LSTM Model
```python
class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        output, (hidden, cell) = self.lstm(embedded)  # hidden: [1, batch_size, hidden_dim]
        return self.fc(hidden.squeeze(0))  # [batch_size, output_dim]

# Hyperparameters
embedding_dim = 32
hidden_dim = 50
output_dim = 1
model = LSTMNet(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
```
- **Structure**: 
  - Embedding layer (10K vocab → 32-dim vectors).
  - LSTM (32-dim input → 50-dim hidden).
  - Linear layer (50 → 1 for binary output).
- **Run**: Initializes the LSTM.

#### Cell 4: Training
```python
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.float().unsqueeze(1).to(device)
        outputs = model(texts)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Plot losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```
- **Features**: 
  - `BCEWithLogitsLoss` for binary classification.
  - Adam optimizer.
  - 10 epochs for sequence model training.
- **Run**: Trains and plots loss curves.

#### Cell 5: Evaluation
```python
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.float().unsqueeze(1).to(device)
        outputs = model(texts)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
```
- **Run**: Expect ~85% accuracy with this setup.

---

### Step 3: Commit to GitHub
- Save `notebook.ipynb` in Cursor.
- In terminal:
  ```bash
  cd Project4_SentimentAnalysis
  git add notebook.ipynb environment.yml requirements.txt README.md
  git commit -m "Complete Project 4: Sentiment Analysis with LSTM"
  git push origin main
  ```

---

### Notes
- **Dataset**: 
  - `torchvision.datasets.IMDB` simplifies loading (downloads to `./data/`).
  - Basic preprocessing (no advanced tokenization like NLTK); sufficient for learning LSTMs.
- **Accuracy**: ~85% is typical for this model. More epochs, larger embeddings, or bidirectional LSTMs could improve it.
- **Progression**: Shifts from CNNs (Project 3) to sequence models, introducing NLP basics.

---

### Try It Out
1. Activate `deeplearningproject4` (`conda activate deeplearningproject4`).
2. Run the notebook in Cursor—check the sample text, loss curves, and accuracy.
3. Verify GitHub has the updates.

Let me know how it goes! If accuracy dips below expectations or you want to refine it, we can adjust. Ready for Project 5 when you’re set!


Thanks for catching that error! The issue is that `torchvision.datasets.IMDB` doesn’t exist—my apologies for the mix-up. The IMDB dataset isn’t part of `torchvision.datasets` (which focuses on vision datasets like MNIST and CIFAR). Instead, it was historically available through `torchtext`, but `torchtext` has evolved, and its datasets API has changed. Since we’re keeping dependencies minimal and avoiding `torchtext` for simplicity, let’s pivot to using PyTorch’s `datasets` from `torch.utils.data` with a manual download of the IMDB dataset from a reliable source (e.g., the original ACL 2011 dataset). This keeps us in pure PyTorch and avoids extra dependencies.

I’ll update Cell 2 to download the IMDB dataset directly, preprocess it ourselves, and integrate it into your workflow. Here’s the fix for Project 4, Cell 2, with explanations.

---

### Fixing the Error: Updated Cell 2 for Project 4

#### Why the Error Happened
- `torchvision.datasets` doesn’t include IMDB—it’s meant for image datasets.
- The IMDB dataset is typically accessed via `torchtext.datasets`, but `torchtext` requires additional setup and has version-specific quirks (e.g., deprecated APIs in newer PyTorch versions).
- We’ll bypass this by manually downloading the dataset and processing it, which aligns with your minimal dependency setup.

#### Revised Cell 2: Load and Preprocess IMDB Dataset
Replace the original Cell 2 with this:

```python
import torch
from torch.utils.data import DataLoader, Dataset
import os
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom IMDB Dataset class
class IMDBDataset(Dataset):
    def __init__(self, root_dir, vocab, max_len=200):
        self.root_dir = root_dir
        self.vocab = vocab
        self.max_len = max_len
        self.data = []
        for label in ['pos', 'neg']:
            label_dir = os.path.join(root_dir, label)
            label_idx = 1 if label == 'pos' else 0
            for filename in os.listdir(label_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                    indices = self.text_to_tensor(text)
                    self.data.append((indices, label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        return text, label

    def text_to_tensor(self, text):
        words = text.lower().split()[:self.max_len]
        indices = [self.vocab.get(word, 1) for word in words]  # 1 is <unk>
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))  # 0 is <pad>
        return torch.tensor(indices, dtype=torch.long)

# Download and extract IMDB dataset
data_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
data_path = './data/aclImdb_v1.tar.gz'
extracted_path = './data/aclImdb'

if not os.path.exists(extracted_path):
    os.makedirs('./data', exist_ok=True)
    print("Downloading IMDB dataset...")
    urllib.request.urlretrieve(data_url, data_path)
    print("Extracting IMDB dataset...")
    with tarfile.open(data_path, 'r:gz') as tar:
        tar.extractall('./data')
    os.remove(data_path)  # Clean up tar file

# Build vocabulary from training data
vocab_size = 10000
max_len = 200
word2idx = {'<pad>': 0, '<unk>': 1}

train_dir = os.path.join(extracted_path, 'train')
for label in ['pos', 'neg']:
    label_dir = os.path.join(train_dir, label)
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            for word in text.lower().split():
                if word not in word2idx and len(word2idx) < vocab_size:
                    word2idx[word] = len(word2idx)

# Load datasets
train_dataset = IMDBDataset(os.path.join(extracted_path, 'train'), word2idx, max_len)
test_dataset = IMDBDataset(os.path.join(extracted_path, 'test'), word2idx, max_len)

# Train/validation split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Visualize a sample
sample_text, sample_label = train_dataset[0]
sample_words = [k for k, v in word2idx.items() if v in sample_text[:20]]
print(f'Sample Label: {"Positive" if sample_label == 1 else "Negative"}')
print(f'Sample Text (first 20 words): {sample_words}')
```

---

### What Changed?
1. **Imports**: 
   - Removed `torchvision.datasets` and added `os`, `tarfile`, and `urllib.request` for manual dataset handling.
   - Kept core PyTorch utilities.

2. **Manual Download**: 
   - Downloads the IMDB dataset (~80MB) from its original source (Stanford AI) and extracts it to `./data/aclImdb/`.
   - Deletes the `.tar.gz` file after extraction to save space.

3. **Custom Dataset**: 
   - `IMDBDataset` class reads `.txt` files from `train/pos`, `train/neg`, `test/pos`, and `test/neg`.
   - Converts text to tensor indices using the pre-built `word2idx` vocabulary.
   - Pads/truncates sequences to `max_len=200`.

4. **Vocabulary**: 
   - Built from training data only (to avoid data leakage), limited to 10K words.
   - `<pad>` (0) and `<unk>` (1) handle padding and unknown words.

5. **Data Split**: 
   - 80/20 train/validation split from the training set (25K total reviews).
   - Test set is separate (25K reviews).

6. **Dependencies**: 
   - No extra packages needed beyond your existing setup (`urllib.request` is in Python’s standard library).

---

### Running the Updated Cell 2
1. **Activate Environment**: 
   - Ensure you’re in `conda activate deeplearningproject4`.
2. **Run in Cursor**: 
   - Add this cell to `notebook.ipynb` and run it.
   - It’ll download and extract the dataset (takes a minute or two, depending on your internet).
   - Outputs a sample label and the first 20 words of a review.
3. **Check Output**: 
   - You should see something like:
     ```
     Downloading IMDB dataset...
     Extracting IMDB dataset...
     Sample Label: Positive
     Sample Text (first 20 words): ['the', 'and', 'a', 'of', ...]
     ```

---

### Rest of Project 4
The other cells (3, 4, 5) from my previous Project 4 response should work as-is with this updated Cell 2, since they rely on the `DataLoader` outputs (`texts` and `labels`) which remain compatible. Here’s a quick check:

- **Cell 3 (LSTM Model)**: Expects `vocab_size=10000`, which matches `len(word2idx)`. No changes needed.
- **Cell 4 (Training)**: Works with `train_loader` and `val_loader` as provided. Run it after Cell 2.
- **Cell 5 (Evaluation)**: Works with `test_loader`. Run it last.

---

### Updates to README (Optional)
Since the dataset is now manually downloaded, update `Project4_SentimentAnalysis/README.md` to reflect this:
```markdown
## Notes
- Dataset downloaded from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz.
- Extracts to `./data/aclImdb/` (not tracked by Git unless added).
```

---

### Commit the Fix
After running all cells successfully:
```bash
cd Project4_SentimentAnalysis
git add notebook.ipynb README.md  # Only if README updated
git commit -m "Fix Project 4 Cell 2: Use manual IMDB dataset download"
git push origin main
```

---

### Notes on Data
- The dataset extracts to `./data/aclImdb/` (~200MB after extraction).
- If `./data/` is in `.gitignore`, it won’t be tracked. Users cloning the repo will need to run Cell 2 to download it themselves.
- Add `data/` to `.gitignore` if you haven’t already:
  ```bash
  echo "data/" >> ../.gitignore
  git add ../.gitignore
  git commit -m "Ignore data/ folder"
  git push origin main
  ```

---

### Try It Out
1. Replace Cell 2 with the new version in `notebook.ipynb`.
2. Run all cells in Cursor—expect ~85% accuracy (might vary slightly).
3. Let me know if you hit any new errors or if the accuracy looks good!

Sorry again for the mix-up with `torchvision`. This solution keeps it simple and dependency-light. Ready for Project 5 once you’re happy with Project 4!