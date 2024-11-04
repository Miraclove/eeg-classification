#------------------------ Header Comment ----------------------#
# File: train.py
# Date: 2024-11-03
# Author: Weizhi Peng
#--------------------------------------------------------------#
# Purpose:
# This script is designed to implement an emotion recognition 
# pipeline using EEG data from the SEED-IV dataset using vanilla pytorch. 
# The pipeline includes data loading, preprocessing, and model training, with 
# a focus on evaluating model performance using 10-fold cross-validation.
# The script employs a Continuous Convolutional Neural Network (CCNN) 
# for emotion classification into four categories based on EEG signals.
#
# Main steps of the pipeline:
# 1. Load and preprocess EEG data from the SEED-IV dataset using TorchEEG.
# 2. Split the dataset into training and testing sets using 10-fold 
#    cross-validation.
# 3. Train a CCNN model on the training set and validate its performance 
#    using a separate validation set.
# 4. Save the best model during training based on validation accuracy.
# 5. Load the best model for each fold and evaluate its performance on 
#    the test set.
# 6. Log and display the test accuracy and average loss for each fold 
#    and across all folds.
#
# Requirements:
# - Python (3.7+)
# - PyTorch
# - TorchEEG library
# - NumPy
# - tqdm (for progress display)
#
# Notes:
# - This script uses a custom logger and training/validation functions.
# - GPU acceleration (CUDA) is leveraged when available, falling back 
#   to CPU if unavailable.
#--------------------------------------------------------------#




from sympy import im
import torch.nn as nn
from torcheeg.models import CCNN

from torcheeg.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from zmq import device
from exp.trainer import train, valid, logger, get_best_device
import torch
from torcheeg import transforms
from torcheeg.datasets import SEEDIVDataset
from torcheeg.datasets.constants.emotion_recognition.seed_iv import (SEED_IV_ADJACENCY_MATRIX,
                                                   SEED_IV_CHANNEL_LOCATION_DICT
                                                   )
from torcheeg.model_selection import KFoldGroupbyTrial
import random
import numpy as np
import os
import torch
from tqdm import tqdm
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

# load dataset
dataset = SEEDIVDataset(io_path=f'../data/SEED_IV/seed_iv',
                    root_path='../data/SEED_IV/eeg_raw_data',
                    offline_transform=transforms.Compose([
                        transforms.BandDifferentialEntropy(),
                        transforms.ToGrid(SEED_IV_CHANNEL_LOCATION_DICT)
                    ]),
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Compose([
                        transforms.Select('emotion'),
                        transforms.Lambda(lambda x: x),
                    ]),
                    num_worker=4)

# split dataset, 10-fold cross validation
k_fold = KFoldGroupbyTrial(n_splits=10,
                        split_path='./.tmp_out/examples_pipeline/split',
                        shuffle=True,
                        random_state=42)

loss_fn = nn.CrossEntropyLoss()
batch_size = 256
device  = get_best_device()
test_accs = []
test_losses = []

for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):
    # initialize model
    model = CCNN(num_classes=4, in_channels=4, grid_size=(9, 9)).to(device)
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4)  # official: weight_decay=5e-1
    # split train and val
    train_dataset, val_dataset = train_test_split(
        train_dataset,
        test_size=0.2,
        split_path=f'./.examples_vanilla_torch/split{i}',
        shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,num_workers=28,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=28,pin_memory=True)

    epochs = 50
    best_val_acc = 0.0
    with tqdm(range(epochs)) as pbar:
        for t in pbar:
            train_loss = train(train_loader, model, loss_fn, optimizer)
            val_acc, val_loss = valid(val_loader, model, loss_fn)
            pbar.set_description(f'Fold {i} Epoch {t+1} Train Loss {train_loss:.4f} Val Acc {val_acc:.4f} Val Loss {val_loss:.4f} Best Val Acc {best_val_acc:.4f}')
            # save the best model based on val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),
                        f'./.examples_vanilla_torch/model{i}.pt')

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load the best model to test on test set
    model.load_state_dict(torch.load(f'./.examples_vanilla_torch/model{i}.pt'))
    test_acc, test_loss = valid(test_loader, model, loss_fn)

    # log the test result
    logger.info(
        f"Test Error {i}: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}"
    )

    test_accs.append(test_acc)
    test_losses.append(test_loss)

# log the average test result on cross-validation datasets
logger.info(
    f"Test Error: \n Accuracy: {100*np.mean(test_accs):>0.1f}%, Avg loss: {np.mean(test_losses):>8f}"
)