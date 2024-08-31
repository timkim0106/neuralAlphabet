import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


# First column is the class label, followed by 784 pixel values (28x28 image)
names = ['class'] + list(range(1, 785))

# csv file
file_path = '/Users/timothykim/Desktop/Python/Neural/A_Z Handwritten Data.csv'
df = pd.read_csv(file_path, header=None, names=names)

# map indices to characters in alphabetical order 
class_mapping = {i: chr(65+i) for i in range(26)}  # A=65 in ASCII

# Convert DataFrame to NumPy array 
data = np.array(df)
np.random.shuffle(data) 


# Development set
dev_size = 1000  # Use 1000 samples for development set
data_dev = data[:dev_size].T
Y_dev, X_dev = data_dev[0], data_dev[1:]
X_dev = X_dev / 255.  # Normalize pixel values from 0 to 1

# Training set
data_train = data[dev_size:].T
Y_train, X_train = data_train[0], data_train[1:]
X_train = X_train / 255.  # Normalize pixel values from 0 to 1

# Convert NumPy arrays to PyTorch tensors
# Reshape the data to (batch_size, channels, height, width) for CNN input
X_train_tensor = torch.tensor(X_train.T, dtype=torch.float32).reshape(-1, 1, 28, 28)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

# Define data augmentation transformations
transform = transforms.Compose([transforms.RandomRotation(10), 
                                transforms.RandomAffine(0, translate=(0.1, 0.1)),  
                                transforms.RandomHorizontalFlip(), 
                                transforms.ToTensor()])

dataset = TensorDataset(X_train_tensor, Y_train_tensor)
# 80% training data
train_size = int(0.8 * len(dataset))  
#20% validation data
val_size = len(dataset) - train_size  

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Neural network architecture
class AdvancedHandwritingRecognitionModel(nn.Module):
    def __init__(self):
        super(AdvancedHandwritingRecognitionModel, self).__init__()
        
        # Two convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Dropout layer 
        self.dropout = nn.Dropout(0.4)
        
        # ReLU activation function
        self.relu = nn.ReLU()
        
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm1d(256)
        
        # Placeholder for the size of the flattened layer
        self._to_linear = None
        # Dummy forward pass to calculate the size of the flattened layer
        self.convs(torch.randn(1, 1, 28, 28))
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 26)  # 26 output classes (A-Z)
  
        self.softmax = nn.Softmax(dim=1)
        
    def convs(self, x):
        # Apply convolutional layers, batch norm, ReLU, and max pooling
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        
        # Calculate the size of the flattened layer if not already done
        if self._to_linear is None:
            self._to_linear = x.numel()
        return x

    def forward(self, x):
        # Apply convolutional layers
        x = self.convs(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply first two fully connected layers, batch norm, and ReLU
        x = self.relu(self.batch_norm3(self.fc1(x)))
        x = self.dropout(x)
   
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Apply final fully connected layer and softmax
        x = self.softmax(self.fc3(x))
        return x

# Initialize the model, loss function, optimizer, and learning rate scheduler
model = AdvancedHandwritingRecognitionModel()
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=25):
    for epoch in range(epochs):
        model.train()  
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Training loop
        for inputs, labels in train_loader:
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels) 
            loss.backward()  
            optimizer.step()  

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
        # Adjust learning rate
        scheduler.step() 
        
        # Calculate average loss and accuracy for this epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_predictions

        # Validation loop
        model.eval()  
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        # Disable gradient computation
        with torch.no_grad():  
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct_predictions += (predicted == labels).sum().item()
                val_total_predictions += labels.size(0)

        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct_predictions / val_total_predictions

        # Print results
        print(f'Epoch {epoch + 1}/{epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Go go
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=25)
