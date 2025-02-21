Glad Project 2 went smoothly! An accuracy of 92%+ is still solid for the improved digit classifier—slight variations can happen due to random weight initialization, data shuffling, or hyperparameter choices. We could tweak it further (e.g., more epochs or layer sizes), but since you’re happy with it, let’s roll into Project 3: **Cats vs. Dogs Image Classifier**. This one steps up to convolutional neural networks (CNNs) using PyTorch, tackling a binary image classification task. We’ll set it up in your repo, keeping the Miniconda + `pip` dual setup, and walk through it in Cursor.

---

### Step 1: Setting Up Project 3 Folder

#### Folder Structure
Adding Project 3 to your repo:
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
│   └── README.md
├── Project5_MusicGenreClassification/
│   └── README.md
├── README.md
└── .gitignore
```

#### Setup Instructions
1. **Navigate to Project 3 Folder**:
   - In Cursor, open `DeepLearningPractice/Project3_CatsVsDogsClassifier/`.
   - `README.md` is already present from the initial setup.

2. **Add `environment.yml` (for Your Miniconda)**:
   - Create `environment.yml` in Cursor:
     ```yaml
     name: deeplearningproject3
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
       - pillow>=9.0.0  # For image processing
       - pip
     ```
   - Set up the environment:
     ```bash
     conda env create -f environment.yml
     conda activate deeplearningproject3
     ```
   - Note: Added `pillow` for image handling (PIL).

3. **Add `requirements.txt` (for Pip Compatibility)**:
   - Create `requirements.txt` in Cursor:
     ```
     torch>=2.0.0
     torchvision>=0.15.0
     numpy>=1.24.0
     matplotlib>=3.7.0
     jupyter>=1.0.0
     pillow>=9.0.0
     ```

4. **Update `README.md`**:
   - Edit `README.md` in Cursor:
     ```markdown
     # Project 3: Cats vs. Dogs Image Classifier

     ## Objective
     Build a convolutional neural network (CNN) in PyTorch to classify images as cats or dogs.

     ## Skills
     - Convolutional layers
     - Pooling layers
     - Data augmentation
     - PyTorch CNN implementation

     ## Steps
     1. Load and preprocess the Cats vs. Dogs dataset.
     2. Define a CNN with convolutional and pooling layers.
     3. Train with data augmentation.
     4. Evaluate on a test set.

     ## Setup
     ### Option 1: Miniconda
     - Create and activate the environment:
       ```bash
       conda env create -f environment.yml
       conda activate deeplearningproject3
       ```
     - Open `notebook.ipynb` in Cursor and run cells.

     ### Option 2: Pip
     - Install dependencies:
       ```bash
       pip install -r requirements.txt
       ```
     - Open `notebook.ipynb` in Cursor or Jupyter Notebook.

     ## Notes
     - Dataset sourced from Kaggle (https://www.kaggle.com/c/dogs-vs-cats/data).
     - Extract `train.zip` to `./data/` (not tracked by Git unless added).
     ```

5. **Create `notebook.ipynb`**:
   - In Cursor, create `notebook.ipynb` in `Project3_CatsVsDogsClassifier/`.

---

### Step 2: Implementing Project 3 in `notebook.ipynb`

Activate your Conda environment (`conda activate deeplearningproject3`), open `notebook.ipynb` in Cursor, and add these cells. We’ll use a subset of the Kaggle Cats vs. Dogs dataset and build a CNN with data augmentation.

#### Cell 1: Imports
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
```
- **Run**: Imports for CNNs and image handling.

#### Cell 2: Load and Preprocess Dataset
```python
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset class
class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        # Subset: First 2000 images to keep it manageable
        self.images = self.images[:2000]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        label = 0 if 'cat' in self.images[idx] else 1  # 0 = cat, 1 = dog
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset (download from Kaggle manually first)
data_dir = './data/train'  # Extract train.zip from Kaggle here
full_dataset = CatsDogsDataset(root_dir=data_dir, transform=None)

# Train/validation/test split (70/15/15)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size]
)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Visualize a sample
images, labels = next(iter(train_loader))
plt.imshow(images[0].permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
plt.title(f'Label: {"Cat" if labels[0].item() == 0 else "Dog"}')
plt.show()
```
- **Setup**: 
  - Download the Cats vs. Dogs dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).
  - Extract `train.zip` to `DeepLearningPractice/Project3_CatsVsDogsClassifier/data/train/`.
  - Uses first 2000 images to keep it light (full dataset is 25K images).
- **Features**: 
  - Custom `Dataset` class for loading images.
  - Augmentation (flips, rotations) for training.
  - Normalizes with ImageNet stats.
  - Splits into train/val/test.
- **Run**: Plots a sample image (undo normalization for display).

#### Cell 3: Define the CNN
```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 3 input channels (RGB), 16 filters
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # Binary output

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = ConvNet().to(device)
```
- **Structure**: 
  - Conv1: 3→16 filters, 3x3, ReLU, MaxPool (64x64→32x32).
  - Conv2: 16→32 filters, 3x3, ReLU, MaxPool (32x32→16x16).
  - FC: Flatten (32*16*16), 64, 1 (sigmoid for binary).
- **Run**: Initializes the CNN.

#### Cell 4: Training with Augmentation
```python
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        outputs = model(images)
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
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
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
  - Uses `BCEWithLogitsLoss` (combines sigmoid + binary cross-entropy).
  - Adam optimizer.
  - 20 epochs to account for CNN complexity.
- **Run**: Trains and plots loss curves.

#### Cell 5: Evaluation
```python
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        outputs = model(images)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
```
- **Run**: Expect ~80-85% accuracy with this small subset and simple CNN.

---

### Step 3: Commit to GitHub
- Save `notebook.ipynb` in Cursor.
- In terminal:
  ```bash
  cd Project3_CatsVsDogsClassifier
  git add notebook.ipynb environment.yml requirements.txt README.md
  git commit -m "Complete Project 3: Cats vs Dogs CNN"
  git push origin main
  ```

---

### Notes
- **Dataset**: 
  - Download `train.zip` from Kaggle and extract to `./data/train/`.
  - Using 2000 images keeps it manageable; full dataset (25K) would take longer but could hit higher accuracy.
  - If `data/` is in `.gitignore`, it won’t be tracked—users must download it themselves.
- **Accuracy**: 80-85% is typical for this setup. More data, epochs, or a deeper CNN could push it higher.
- **Progression**: Moves from dense layers (Projects 1-2) to CNNs, introducing image processing and augmentation.

---

### Try It Out
1. Activate `deeplearningproject3` (`conda activate deeplearningproject3`).
2. Extract the dataset to `./data/train/`.
3. Run the notebook in Cursor—check the sample image, loss curves, and accuracy.
4. Verify GitHub has the updates.

Let me know how it goes! If accuracy is lower than expected or you want to tweak it, we can adjust. Ready for Project 4 when you are!