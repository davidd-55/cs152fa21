#!/usr/bin/env python

from argparse import ArgumentParser
from utils import get_mnist_data_loaders, DataLoaderProgress
from fastprogress.fastprogress import master_bar, progress_bar
import torch
import torch.nn as nn


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

    loss, num_correct = 0, 0

    # Tell pytorch to stop updating gradients when executing the following
    with torch.no_grad():

        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

            # Compute the model output
            output = model(X)

            # Compute loss
            loss += criterion(output, Y).item()

            # Compute the number of correctly classified examples
            num_correct += (output.argmax(1) == Y).type(torch.float).sum().item()
            

        loss /= num_batches
        accuracy = num_correct / N

    message = "Initial" if epoch == 0 else f"Epoch {epoch:>2}/{num_epochs}:"
    message += f" accuracy={100*accuracy:5.2f}%"
    message += f" and loss={loss:.3f}"
    mb.write(message)


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
    train_loader, valid_loader = get_mnist_data_loaders(args.mnist, args.batch_size, 0)

    batch_X, batch_Y = next(iter(train_loader))

    # Neural network model
    nx = batch_X.shape[1:].numel()
    ny = int(torch.unique(batch_Y).shape[0])
    layer_sizes = (nx, 10, 5, ny)

    model = nn.NeuralNetwork(layer_sizes).to(device)

    # CrossEntropyLoss criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(
        model, criterion, optimizer, train_loader, valid_loader, device, args.num_epochs
    )


if __name__ == "__main__":
    main()
