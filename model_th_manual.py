import ray
from ray import tune
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
import random
import numpy as np
from filelock import FileLock
import torchmetrics


def objective(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mnist_train = dsets.MNIST(
        root="MNIST_data/",  
        train=True,  
        transform=transforms.ToTensor(),  
        download=True,
    )

    mnist_test = dsets.MNIST(
        root="MNIST_data/",  
        train=False,  
        transform=transforms.ToTensor(), 
        download=True,
    )

    x_train, x_val, y_train, y_val = train_test_split(
        mnist_train.data, mnist_train.targets, test_size=0.3, random_state=777
    )

    train_data_loader = DataLoader(
        dataset=TensorDataset(x_train, y_train),
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )

    val_data_loader = DataLoader(
        dataset=TensorDataset(x_val, y_val),
        batch_size=64,
        shuffle=True,
        drop_last=True,
    )

    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.flatten = nn.Flatten()
            self.Dense1 = nn.Linear(784, config["neuron1"])
            self.Dense2 = nn.Linear(config["neuron1"], config["neuron2"])
            self.output = nn.Linear(config["neuron1"], 10)

            if config["activation"] == "relu":
                self.activation = nn.ReLU()
            if config["activation"] == "tanh":
                self.activation = nn.Tanh()

        def forward(self, x):
            x = self.flatten(x)
            print(x)
            x = self.activation(self.Dense1(x))
            x = self.activation(self.Dense2(x))
            x = F.log_softmax(x)
            return x

    train_loss = nn.CrossEntropyLoss().to(device)
    train_acc = torchmetrics.Accuracy().to(device)
    test_acc = torchmetrics.Accuracy().to(device)

    model = Net().to(device)

    print(model)
    if config["optimizers"] == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters())
    if config["optimizers"] == "adam":
        optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(config["training_iteration"]):

        train_acces = []
        val_acces = []
        losses = []
        for images, labels in train_data_loader:
            optimizer.zero_grad()
            predicted = model(images.to(device).float())
            loss = train_loss(predicted, labels.to(device))
            loss.backward()
            acc = train_acc(predicted, labels.to(device))
            optimizer.step()
            losses.append(loss.item())
            train_acces.append(acc)

        loss_print = sum(losses) / len(train_data_loader)
        acc_print = sum(train_acces) / len(train_data_loader)

        model.eval()
        for test_images, test_labels in val_data_loader:
            with torch.no_grad():
                predicted = model(test_images.to(device).float())
                val_acc = test_acc(predicted, test_labels.to(device))
                val_acces.append(val_acc)

        val_acc_print = sum(val_acces) / len(val_data_loader)

        print(val_acc_print)

    tune.report(mean_accuracy=val_acc_print.item())
