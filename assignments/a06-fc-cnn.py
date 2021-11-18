#!/usr/bin/env python

from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, alexnet
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, ToTensor

from fastprogress.fastprogress import master_bar, progress_bar


class DataLoaderProgress(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.length = len(dataloader)

    def __iter__(self):
        return zip(range(self.length), self.dataloader)

    def __len__(self):
        return self.length


class NN_FC_CrossEntropy(nn.Module):
    def __init__(self, layer_sizes):
        super(NN_FC_CrossEntropy, self).__init__()

        first_layer = nn.Flatten()
        middle_layers = [
            nn.Sequential(nn.Linear(nlminus1, nl), nn.ReLU())
            for nl, nlminus1 in zip(layer_sizes[1:-1], layer_sizes)
        ]
        last_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        all_layers = [first_layer] + middle_layers + [last_layer]

        self.layers = nn.Sequential(*all_layers)

    def forward(self, X):
        return self.layers(X)


def get_cifar10_data_loaders(path, batch_size, valid_batch_size):

    std = (0.4941, 0.4870, 0.5232)
    mean = (-0.0172, -0.0357, -0.1069)
    image_xforms = Compose([ToTensor(), Normalize(mean, std)])
    
    train_dataset = CIFAR10(root=path, train=True, download=True, transform=image_xforms)
    
    tbs = len(train_dataset) if batch_size == 0 else batch_size
    train_loader = DataLoader(train_dataset, batch_size=tbs, shuffle=True)

    valid_dataset = CIFAR10(root=path, train=False, download=True, transform=image_xforms)
    
    vbs = len(valid_dataset) if valid_batch_size == 0 else valid_batch_size
    valid_loader = DataLoader(valid_dataset, batch_size=vbs, shuffle=True)

    return train_loader, valid_loader


def train_one_epoch(dataloader, model, criterion, optimizer, device, mb):

    # Put the model into training mode
    model.train()

    size = len(dataloader.dataset)

    # Loop over the data using the progress_bar utility
    for batch, (X, Y) in progress_bar(DataLoaderProgress(dataloader), parent=mb):
        X, Y = X.to(device), Y.to(device)

        # Compute model output and then loss
        output = model(X)
        loss = criterion(output, Y)

        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print batch info
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validate(dataloader, model, criterion, device, epoch, num_epochs, mb):

    # Put the model into validation/evaluation mode
    model.eval()

    N = len(dataloader.dataset)
    num_batches = len(dataloader)

    valid_loss, num_correct = 0, 0

    # Tell pytorch to stop updating gradients when executing the following
    with torch.no_grad():

        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

            # Compute the model output
            output = model(X)

            # Compute loss
            valid_loss += criterion(output, Y).item()

            # Compute the number of correctly classified examples
            num_correct += (output.argmax(1) == Y).type(torch.float).sum().item()
            

        valid_loss /= num_batches
        valid_accuracy = num_correct / N

    mb.write(
        f"{epoch:>3}: validation accuracy={(100*valid_accuracy):5.2f}% and loss={valid_loss:.3f}"
    )
    return valid_loss, valid_accuracy


def train(model, criterion, optimizer, train_loader, valid_loader, device, num_epochs):

    mb = master_bar(range(num_epochs))

    validate(valid_loader, model, criterion, device, 0, num_epochs, mb)

    for epoch in mb:
        train_one_epoch(train_loader, model, criterion, optimizer, device, mb)
        validate(valid_loader, model, criterion, device, epoch + 1, num_epochs, mb)


def main():

    aparser = ArgumentParser("Train a neural network on the MNIST dataset.")
    aparser.add_argument("mnist", type=str, help="Path to store/find the MNIST dataset")
    aparser.add_argument("--num_epochs", type=int, default=10)
    aparser.add_argument("--batch_size", type=int, default=128)
    aparser.add_argument("--learning_rate", type=float, default=0.1)
    aparser.add_argument("--seed", action="store_true")
    aparser.add_argument("--gpu", action="store_true")

    args = aparser.parse_args()

    # Set the random number generator seed if one is provided
    if args.seed:
        torch.manual_seed(args.seed)

    # Use GPU if requested and available
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"'{device}' selected as hardware device.")

    # Get data loaders
    train_loader, valid_loader = get_cifar10_data_loaders(args.cifar10, args.batch_size, 0)

    batch_X, batch_Y = next(iter(train_loader))

    # Neural network model
    nx = batch_X.shape[1:].numel()
    ny = int(torch.unique(batch_Y).shape[0])
    layer_sizes = (nx, 100, 75, 50, ny)


    #model = NN_FC_CrossEntropy(layer_sizes).to(device)

    model = resnet18()

    #model = alexnet()
    
    model.fc = nn.Linear(in_features=784, out_features=10, bias=True)
    model.to(device)

    # CrossEntropyLoss criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(
        model, criterion, optimizer, train_loader, valid_loader, device, args.num_epochs
    )


if __name__ == "__main__":
    main()
