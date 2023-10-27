import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import optuna
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize image to [-1, 1]

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# dataloader for batch training (mini-batch gradient descent)
batch_size=32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
# 10 classes in total
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" torch.cuda.is_available(), {torch.cuda.is_available()}")
print(f" Device index: {torch.cuda.current_device()}")

class CNN(nn.Module):
    def __init__(self, n_filters_conv1=32, n_filters_conv2=64, fc1_units=64, fc2_units=64, dropout_prob=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, n_filters_conv1, 5) 
        self.bn1 = nn.BatchNorm2d(n_filters_conv1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_filters_conv1, n_filters_conv2, 5)
        self.bn2 = nn.BatchNorm2d(n_filters_conv2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(n_filters_conv2*5*5, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 10)
        

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1) 
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def objective(trial):
   # Define hyperparameters using the trial object
    n_filters_conv1 = trial.suggest_int('n_filters_conv1', 16, 128, log=True)
    n_filters_conv2 = trial.suggest_int('n_filters_conv2', 64, 512, log= True)
    fc1_units = trial.suggest_int('fc1_units', 64, 256)
    fc2_units = trial.suggest_int('fc2_units', 64, 256)
    lr = trial.suggest_loguniform('lr', 1e-3, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        # Data loading
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize image to [-1, 1]
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    model = CNN(n_filters_conv1, n_filters_conv2, fc1_units, fc2_units)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training the model
    for epoch in range(10): # train for 10 epochs as an example
        for data, target in trainloader:
            data, target = data.to(device),target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Evaluating the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device),target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return accuracy

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
study = optuna.create_study(direction='maximize') # maximize accuracy
study.optimize(objective, n_trials=10) # search across 50 different hyperparameter combinations

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")