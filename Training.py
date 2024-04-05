import numpy as np
import keras
from scipy import io as spio
import tensorflow as tf
import os
import torch
from LeNet5 import LeNet5
from torch.optim import SGD


EMNIST = spio.loadmat(os.path.join(os.getcwd(), 
                                   "Desktop",
                                   "Machine learning and Big Data processing",
                                   "Projet",
                                   "matlab", "emnist-mnist.mat"))

x_train = EMNIST["dataset"][0][0][0][0][0][0]
y_train = EMNIST["dataset"][0][0][0][0][0][1]

x_test = EMNIST['dataset'][0][0][1][0][0][0]
y_test = EMNIST['dataset'][0][0][1][0][0][1]

# Scaling data
x_train_scaled = (x_train - np.mean(x_train))/np.std(x_train)

x_test_scaled = (x_test - np.mean(x_train))/np.std(x_train)

nb_classes = 10

y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)


x_train_scaled = x_train.reshape(-1, 28, 28, 1)
x_test_scaled = x_test.reshape(-1, 28, 28, 1)

x_train_padded = np.array(tf.pad(tensor = x_train_scaled, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]]))
x_test_padded = np.array(tf.pad(tensor = x_test_scaled, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]]))

# Pytorch needs a special format
x_train_tensor = torch.tensor(x_train_padded, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test_padded, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


num_epochs = 20  
batch_size = 300  

model = LeNet5()

#Loss 
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = SGD(model.parameters(), lr=0.01)

# Training
for epoch in range(num_epochs):
    # Shuffle the data at the begining of an epoch
    
    indices = np.arange(len(x_train_tensor))
    np.random.shuffle(indices)
    total_correct_predictions = 0
    total_samples = 0
    
    for i in range(0, len(x_train), batch_size):
        # Select batch by batch
        batch_indices = indices[i:i+batch_size]
        inputs = x_train_tensor[batch_indices].permute(0, 3, 1, 2)
    
        labels = y_train_tensor[batch_indices]

        # Forward propagation
        outputs = model(inputs)
        
        # Compute the loss
        loss = criterion(outputs, labels)

        model.eval()
        _, predicted_labels = torch.max(outputs, 1)
        _, target_labels = torch.max(labels, 1)
        batch_correct_predictions = (predicted_labels == target_labels).sum().item()
        
        # Accumulate the total number of correct predictions and total samples
        total_correct_predictions += batch_correct_predictions
        total_samples += labels.size(0)
        model.train()

        # Back-propagation and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate accuracy at the end of the epoch
    epoch_accuracy = total_correct_predictions / total_samples
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy:.4f}')