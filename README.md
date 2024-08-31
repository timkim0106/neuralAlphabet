# Handwritten Alphabet Recognition

I practiced building a Convolutional Neural Network with PyTorch to recognize handwritten alphabets from a Kaggle A-Z dataset. 

From this project, I practiced

- loading and preprocessing data
- Designing a CNN model with batch normalization and dropout
- Training and evaluating the model
- Using data augmentation techniques to improve model performance

The model architecture consists of 
  - Two convolutional layers
  - Max pooling
  - Batch normalization
  - Dropout
  - Fully connected layers

- Data augmentation for robust training
- Learning rate scheduling for optimized training

The training parameters I used were:
- Epochs: 25
- Batch Size: 64
- Learning Rate: 0.001
- Weight Decay: 1e-4
- Learning Rate Scheduler: StepLR (step_size=5, gamma=0.5)

After training, I produced a validation accuracy of 95%.

