import optuna
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize image to [-1, 1]

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# 10 classes in total
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 


# load test data (note that the data has been transformed already)
test_images = torch.load('test_image.pt')

# specify the GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#to try ResNet models with different number of layers
def create_resnet(num_layers, dropout_rate):
    if num_layers == 18:
        net = models.resnet18(pretrained=False,num_classes=10).to(device)
    elif num_layers == 34:
        net = models.resnet34(pretrained=False,num_classes=10).to(device)
    elif num_layers == 50:
        net = models.resnet50(pretrained=False,num_classes=10).to(device)
    else:
        raise ValueError("Unsupported number of layers")

    num_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, 10)
    )
    return net

#define the tuning function 
 
 def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True)
    num_layers = trial.suggest_categorical('num_layers', [18, 34, 50])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    net = create_resnet(num_layers, dropout_rate).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    best_acc = 0.0

    for epoch in range(15):  #after several attempts, it seems like the loss become stable after 10 epochs
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Accuracy: {acc}%")
        if acc > best_acc:
            best_acc = acc

    return best_acc  

# Optuna optimization
study = optuna.create_study(direction='maximize')  
study.optimize(objective, n_trials=1)  

# Results
print(f'Best trial: score {study.best_value}, params {study.best_params}')