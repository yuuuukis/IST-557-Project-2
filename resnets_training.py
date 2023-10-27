import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd

# load and transfor training data from standard source
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize image to [-1, 1]

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# dataloader for batch training (mini-batch gradient descent)
batch_size=32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
# 10 classes in total
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 


test_images = torch.load('test_image.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision.models as models

net = models.resnet18(pretrained=False, num_classes=10).to(device)

import torch.optim as optim
import torch.nn as nn


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

best_acc = 0.0  # track the best accuracy

# Training loop
for epoch in range(100):  # loop over the dataset multiple times (e.g., 20 epochs)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {running_loss / len(trainloader)}")
    
    
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

    # Check if this model has the best accuracy
    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), 'best_model.pth')  # save the best model
        
print('Finished Training')