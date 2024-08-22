import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
#

# Define column names for the dataset
names = ['class']
for id in range(1, 785):
    names.append(id)

# Load the dataset from a CSV file
file_path = '/Users/timothykim/Desktop/Python/Neural/A_Z Handwritten Data.csv'
file_path1 = '/kaggle/input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv'
df = pd.read_csv(file_path, header=None, names=names)

# Create a mapping of class indices to alphabet characters
class_mapping = {}
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i in range(len(alphabets)):
    class_mapping[i] = alphabets[i]
class_mapping

# Convert DataFrame to NumPy array
data = np.array(df)
m, n = data.shape

# Shuffle the data to randomize the training and validation sets
np.random.shuffle(data)

# Split the data into development and training sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.  # Normalize pixel values to the range [0, 1]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.  # Normalize pixel values to the range [0, 1]
_, m_train = X_train.shape

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train.T, dtype=torch.float32).reshape(-1, 1, 28, 28)  # Reshape for CNN
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

# Data Augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Creating a dataset and splitting it
dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # Remaining 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)



# Define an advanced neural network with PyTorch
class AdvancedHandwritingRecognitionModel(nn.Module):
    def __init__(self):
        super(AdvancedHandwritingRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Convolutional layer with 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Convolutional layer with 64 filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling layer
        self.dropout = nn.Dropout(0.4)  # Dropout layer with 40% dropout rate
        self.relu = nn.ReLU()  # ReLU activation function
        
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm1d(256)
        
        # Calculate the size of the output from conv layers
        self._to_linear = None
        self.convs(torch.randn(1, 1, 28, 28))  # Dummy pass to get the _to_linear value
        
        self.fc1 = nn.Linear(self._to_linear, 256)  # Fully connected layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Fully connected layer with 128 units
        self.fc3 = nn.Linear(128, 26)  # Output layer with 26 units (one for each class)
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multi-class classification
        
    def convs(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))  # Convolution -> ReLU -> BatchNorm -> MaxPool
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))  # Convolution -> ReLU -> BatchNorm -> MaxPool
        if self._to_linear is None:
            self._to_linear = x.numel()  # Compute the size of the flattened layer
        return x

    def forward(self, x):
        x = self.convs(x)  # Apply convolutional layers
        x = x.view(x.size(0), -1)  # Flattening
        x = self.relu(self.batch_norm3(self.fc1(x)))  # Fully connected layer -> ReLU -> BatchNorm
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))  # Fully connected layer -> ReLU
        x = self.dropout(x)  # Apply dropout
        x = self.softmax(self.fc3(x))  # Output layer with softmax activation
        return x

model = AdvancedHandwritingRecognitionModel()

# Define loss function and optimizer with a scheduler
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adam optimizer with weight decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Learning rate scheduler

# Training the model with learning rate scheduler
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=25):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters
            
            running_loss += loss.item() * inputs.size(0)  # Accumulate loss
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
            total_predictions += labels.size(0)  # Count total predictions

        scheduler.step()  # Update learning rate
        
        epoch_loss = running_loss / len(train_loader.dataset)  # Compute average loss
        epoch_accuracy = correct_predictions / total_predictions  # Compute training accuracy

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                val_loss += loss.item() * inputs.size(0)  # Accumulate validation loss
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                val_correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
                val_total_predictions += labels.size(0)  # Count total predictions

        val_loss = val_loss / len(val_loader.dataset)  # Compute average validation loss
        val_accuracy = val_correct_predictions / val_total_predictions  # Compute validation accuracy

        print(f'Epoch {epoch + 1}/{epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=25)


# 