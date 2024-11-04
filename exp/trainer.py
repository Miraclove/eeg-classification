#------------------------ Header Comment ----------------------#
# File: trainer.py
# Date: 2024-11-03
# Author: Weizhi Peng
#--------------------------------------------------------------#
# Purpose:
# This script provides utility functions and configurations for training 
# and validating machine learning models using vanilla PyTorch. It includes 
# functions for model training, validation, and logging setup, as well as 
# a utility function to determine the best available device for computation.
#
# Main Components:
# 1. Logger setup for training logs, including both console and file handlers.
# 2. `get_best_device()` function to determine and return the best available 
#    computation device (CUDA, MPS, or CPU).
# 3. `train()` function for training a model using a specified DataLoader, 
#    loss function, and optimizer.
# 4. `valid()` function for validating a model's performance on a validation 
#    dataset and returning accuracy and average loss.
#
# Requirements:
# - Python (3.7+)
# - PyTorch
# - logging library (built-in)
#
# Notes:
# - The script is designed to handle training with GPU acceleration when 
#   available, falling back to CPU if necessary.
# - This code is intended for integration with training pipelines and can 
#   be adapted for specific use cases.
#--------------------------------------------------------------#

import torch
import os
import time
import logging


os.makedirs('./.train/log', exist_ok=True)
logger = logging.getLogger('Training models with vanilla PyTorch')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join('./.train/log', f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_best_device():
    """
    Determines the best available device for computation.

    This function checks for the availability of CUDA (NVIDIA GPUs) and MPS (Apple Silicon GPUs)
    and returns the appropriate device string. If neither CUDA nor MPS is available, it defaults to CPU.

    Returns:
        str: The best available device ('cuda', 'mps', or 'cpu').
    """
    # get accelerator
    if torch.cuda.is_available():
        print('cuda is available')
        device = 'cuda'
    elif torch.has_mps:
        print('mps is available')
        device = 'mps'
    else:
        print('cuda is not available, use cpu')
        device = 'cpu'
    return device

device = get_best_device()
# training process
def train(dataloader, model, loss_fn, optimizer):
    """
    Trains the given model using the provided dataloader, loss function, and optimizer.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The neural network model to be trained.
        loss_fn (torch.nn.Module): The loss function to be used for training.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating model parameters.

    Returns:
        float: The final loss value after training.
    """
    size = len(dataloader.dataset)
    record_step = int(len(dataloader) / 10)

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % record_step == 0:
            loss, current = loss.item(), batch_idx * len(X)
            # logger.info(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


# validation process
def valid(dataloader, model, loss_fn):
    """
    Evaluate the model on the validation dataset.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): The neural network model to be evaluated.
        loss_fn (torch.nn.Module): The loss function used to compute the loss.

    Returns:
        tuple: A tuple containing:
            - correct (float): The accuracy of the model on the validation dataset.
            - loss (float): The average loss of the model on the validation dataset.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    # logger.info(
    #     f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n"
    # )
    return correct, loss