from argparse import ArgumentParser
from utils import get_mnist_data_loaders, DataLoaderProgress
from fastprogress.fastprogress import master_bar, progress_bar
import torch
import torch.nn as nn

def train_one_epoch(dataloader, model, criterion, optimizer, device, mb):
    print("Optimization: None")
    
    # Put the model into training mode
    model.train()

    # Loop over the data using the progress_bar utility
    for _, (X, Y) in progress_bar(DataLoaderProgress(dataloader), parent=mb):
        X, Y = X.to(device), Y.to(device)

        # Compute model output and then loss
        output = model(X)
        loss = criterion(output, Y)

        # TODO:
        # - zero-out gradients
        optimizer.zero_grad()        
        # - compute new gradients
        loss.backward()
        # - update paramters
        optimizer.step()

def train_one_epoch_momentum(
    dataloader, model, criterion, learning_rate, weight_decay, momentum, device, mb
):
    print("Optimization: Momentum")
    
    if not hasattr(model, 'momentum_grads'):
        model.momentum_grads = [torch.zeros_like(p) for p in model.parameters()]
    
    model.train()

    num_batches = len(dataloader)
    dataiter = iter(dataloader)

    for batch in progress_bar(range(num_batches), parent=mb):

        X, Y = next(dataiter)
        X, Y = X.to(device), Y.to(device)

        output = model(X)

        loss = criterion(output, Y)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param, grad in zip(model.parameters(), model.momentum_grads):
                grad.set_(momentum * grad + (1 - momentum) * param.grad)
                param -= learning_rate * grad + weight_decay * param

def train_one_epoch_adagrad(
    dataloader, model, criterion, learning_rate, decay_rate, device, mb
):
    print("Optimization: Adagrad")

    if not hasattr(model, "sum_square_grads"):
        model.sum_square_grads = [torch.zeros_like(p) for p in model.parameters()]
        model.ms = [torch.zeros_like(p) for p in model.parameters()]
        model.vs = [torch.zeros_like(p) for p in model.parameters()]
        model.t = 1

    model.train()

    num_batches = len(dataloader)
    dataiter = iter(dataloader)

    for batch in progress_bar(range(num_batches), parent=mb):

        X, Y = next(dataiter)
        X, Y = X.to(device), Y.to(device)

        output = model(X)

        loss = criterion(output, Y)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param, G in zip(model.parameters(), model.sum_square_grads):
                # Adagrad
                G.set_(G + param.grad * param.grad)
                param -= learning_rate * param.grad / (torch.sqrt(G) + 1e-8)

def train_one_epoch_adam(
    dataloader, model, criterion, learning_rate, device, mb
):
    print("Optimization: Adam")
    
    betas = (0.9, 0.999)

    if not hasattr(model, "sum_square_grads"):
        model.sum_square_grads = [torch.zeros_like(p) for p in model.parameters()]
        model.ms = [torch.zeros_like(p) for p in model.parameters()]
        model.vs = [torch.zeros_like(p) for p in model.parameters()]
        model.t = 1

    model.train()

    num_batches = len(dataloader)
    dataiter = iter(dataloader)

    for batch in progress_bar(range(num_batches), parent=mb):

        X, Y = next(dataiter)
        X, Y = X.to(device), Y.to(device)

        output = model(X)

        loss = criterion(output, Y)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param, m, v in zip(model.parameters(), model.ms, model.vs):
                # Adam
                beta1, beta2 = betas
                m.set_(beta1 * m + (1 - beta1) * param.grad)
                v.set_(beta2 * v + (1 - beta2) * param.grad * param.grad)

                mt = m / (1 - beta1 ** model.t)
                vt = v / (1 - beta2 ** model.t)

                param -= learning_rate * mt / (torch.sqrt(vt) + 1e-8)

                model.t += 1

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

            # TODO:
            # - compute loss
            loss += criterion(output, Y).item()
            # - compute the number of correctly classified examples
            num_correct += (output.argmax(1) == Y).type(torch.float).sum().item()


        loss /= num_batches
        accuracy = num_correct / N

    message = "Initial" if epoch == 0 else f"Epoch {epoch:>2}/{num_epochs}:"
    message += f" accuracy={100*accuracy:5.2f}%"
    message += f" and loss={loss:.3f}"
    mb.write(message)

def train(model, criterion, optimizer, train_loader, valid_loader, device, num_epochs, learning_rate=1e-2, weight_decay=1e-3, momentum=0.9):

    mb = master_bar(range(num_epochs))

    validate(valid_loader, model, criterion, device, 0, num_epochs, mb)
    #     dataloader, model, criterion, learning_rate, weight_decay, momentum, device, mb
    
    for epoch in mb:
        # train_one_epoch(train_loader, model, criterion, optimizer, device, mb)        
        # train_one_epoch_momentum(train_loader, model, criterion, learning_rate, weight_decay, momentum, device, mb)
        # train_one_epoch_adam(train_loader, model, criterion, learning_rate, device, mb,)        
        train_one_epoch_adagrad(train_loader, model, criterion, learning_rate, weight_decay, device, mb,)        
        validate(valid_loader, model, criterion, device, epoch + 1, num_epochs, mb)

class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(NeuralNetwork, self).__init__()

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

    # Get data loaders
    train_loader, valid_loader = get_mnist_data_loaders(args.mnist, args.batch_size, 0)

    # TODO: create a new model
    # Your model can be as complex or simple as you'd like. It must work
    # with the other parts of this script.
    nx = train_loader.dataset.data.shape[1:].numel()
    ny = len(train_loader.dataset.classes)
    layer_sizes = (nx, 512, 50, ny)
    model = NeuralNetwork(layer_sizes).to(device)

    # TODO:
    # - create a CrossEntropyLoss criterion
    # - create an optimizer of your choice
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    train(
        model, criterion, optimizer, train_loader, valid_loader, device, args.num_epochs
    )

if __name__ == "__main__":
    main()