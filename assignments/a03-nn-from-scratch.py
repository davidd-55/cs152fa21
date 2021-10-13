#!/usr/bin/env python3

import torch
from torch import Tensor
from typing import Tuple


def initialize_parameters(
    n0: int, n1: int, n2: int, scale: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initialize parameters for a 2-layer neural network.

    Args:
        n0 (int): Number of input features (aka nx)
        n1 (int): Number of neurons in layer 1
        n2 (int): Number of output neurons
        scale (float): Scaling factor for parameters

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: weights and biases for 2 layers
            W1 : (n1, n0)
            b1 : (n1)
            W2 : (n2, n1)
            b2 : (n2)
    """
    
    W1 = torch.randn(n1, n0) * scale
    b1 = torch.randn(n1) * scale
    W2 = torch.randn(n2, n1) * scale
    b2 = torch.randn(n2) * scale
    
    return W1, b1, W2, b2


def linear(A, W, b):
    return A @ W.T + b


def sigmoid(Z):
    return 1 / (1 + torch.exp(-Z))


def forward_propagation(
    A0: Tensor, W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor
) -> Tuple[Tensor, Tensor]:
    """Compute the output of a 2-layer neural network.

    Args:
        A0 (Tensor): (N, n0) input matrix (aka X)
        W1 (Tensor): (n1, n0) layer 1 weight matrix
        b1 (Tensor): (n1) layer 1 bias matrix
        W2 (Tensor): (n2, n1) layer 2 weight matrix
        b2 (Tensor): (n2) layer 2 bias matrix

    Returns:
        Tuple[Tensor, Tensor]: outputs for layers 1 (N, n1) and 2 (N, n2)
    """
    Z1 = linear(A0, W1, b1)
    A1 = sigmoid(Z1)

    Z2 = linear(A1, W2, b2)
    A2 = sigmoid(Z2)
    
    return A1, A2


def sigmoid_to_binary(A2: Tensor) -> Tensor:
    """Convert the output of a final layer sigmoids to zeros and ones.

    Args:
        A2 (Tensor): (N, n2) output of the network

    Returns:
        Tensor: binary predictions of a 2-layer neural network
    """

    return A2.apply_(lambda x: 1 if x >= .5 else 0).int()


def backward_propagation(
    A0: Tensor, A1: Tensor, A2: Tensor, Y: Tensor, W2: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute gradients of a 2-layer neural network's parameters.

    Args:
        A0 (Tensor): (N, n0) input matrix (aka X)
        A1 (Tensor): (N, n1) output of layer 1 from forward propagation
        A2 (Tensor): (N, n2) output of layer 2 from forward propagation (aka Yhat)
        Y (Tensor): (N, n2) correct targets (aka targets)
        W2 (Tensor): (n2, n1) weight matrix

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: gradients for weights and biases
    """
    # TODO: compute and return gradients


def update_parameters(
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    dW1: Tensor,
    db1: Tensor,
    dW2: Tensor,
    db2: Tensor,
    lr: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Update parameters of a 2-layer neural network.

    Args:
        W1 (Tensor): (n1, n0) weight matrix
        b1 (Tensor): (n1) bias matrix)
        W2 (Tensor): (n2, n1) weight matrix)
        b2 (Tensor): (n2) bias matrix
        dW1 (Tensor): (n1, n0) gradient matrix
        db1 (Tensor): (n1) gradient matrix)
        dW2 (Tensor): (n2, n1) gradient matrix)
        db2 (Tensor): (n2) gradient matrix
        lr (float): learning rate

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: updated network parameters
    """
    # TODO: Update and return parameters


def compute_loss(A2: Tensor, Y: Tensor) -> Tensor:
    """Compute mean loss using binary cross entropy loss.

    Args:
        A2 (Tensor): (N, n2) matrix of neural network output values (aka Yhat)
        Y (Tensor): (N, n2) correct targets (aka targets)

    Returns:
        Tensor: computed loss
    """
    
    return -(Y * torch.log(A2) + (1 - Y) * torch.log(1 - A2))


def train_2layer(
    X: Tensor,
    Y: Tensor,
    num_hidden: int,
    param_scale: float,
    num_epochs: int,
    learning_rate: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """A function for performing batch gradient descent with a 2-layer network.

    Args:
        X (Tensor): (N, nx) matrix of input features
        Y (Tensor): (N, ny) matrix of correct targets (aka targets)
        num_hidden (int): number of neurons in layer 1
        param_scale (float): scaling factor for initializing parameters
        num_epochs (int): number of training passes through all data
        learning_rate (float): learning rate

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: learned parameters of a 2-layer neural network
    """
    # TODO: implement this function
    # Steps:
    # 1. create and initialize parameters
    # 2. loop
    #   1. compute outputs with forward propagation
    #   2. compute loss (for analysis)
    #   3. compute gradients with backward propagation
    #   4. update parameters
    # 3. return final parameters


if __name__ == "__main__":
    T1 = torch.tensor([[.6, .4], [.2, .99]])
    print(T1)
    T1_to_binary = sigmoid_to_binary(T1)
    print(T1_to_binary)

    #print(initialize_parameters(2,2,2,1))