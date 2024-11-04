#------------------------ Header Comment ----------------------#
# File: tscnn-training.py
# Date: 2024-10-14
# Author: Weizhi Peng
#--------------------------------------------------------------#
# Purpose:
# This script is designed to implement an emotion recognition 
# pipeline using EEG data from the SEED-IV dataset. The pipeline 
# involves data preprocessing, model training, and evaluation using 
# 10-fold cross-validation. It employs a convolutional neural network 
# (CCNN) for emotion classification into four categories based on 
# EEG signals.
#
# Main steps of the pipeline:
# 1. Load and preprocess EEG data from SEED-IV dataset.
# 2. Apply 10-fold cross-validation for model evaluation.
# 3. Train a CCNN model for emotion classification.
# 4. Use early stopping and model checkpointing during training.
# 5. Evaluate the model and display test accuracy for each fold.
#
# Requirements:
# - PyTorch
# - PyTorch Lightning
# - TorchEEG library
#
# Notes:
# This script leverages GPU acceleration (CUDA or MPS) when available, 
# falling back to CPU if neither is accessible.
#--------------------------------------------------------------#

from torcheeg import transforms
from torcheeg.datasets import SEEDIVDataset
from torcheeg.datasets.constants.emotion_recognition.seed_iv import (SEED_IV_ADJACENCY_MATRIX,
                                                   SEED_IV_CHANNEL_LOCATION_DICT
                                                   )
from torch.utils.data import DataLoader
from model.tscnn import TSCNN
from torcheeg.trainers import ClassifierTrainer
import torch
from torcheeg.model_selection import KFoldGroupbyTrial
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


if __name__ == '__main__':
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


    # training model
    batch_size = 64
    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        # create dataloader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # create model
        model = TSCNN(num_classes=4, in_channels=4, grid_size=(9, 9))

        # create trainer   
        trainer = ClassifierTrainer(model=model,
                                    num_classes=4,
                                    lr=1e-4,
                                    weight_decay=1e-4,
                                    accelerator=device)
        
        # create callbacks, early stopping and model checkpoint
        early_stopping = EarlyStopping('val_loss',patience=3, mode='min')
        current_fold_str = f'fold_{i}'
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath='../model/seediv/tscnn',
                                              filename='tscnn-seediv-'+current_fold_str+'-epoch_{epoch:02d}-val_loss_{val_loss:.4f}-val_accuracy_{val_accuracy:.4f}',
                                              auto_insert_metric_name=False,
                                              mode='min'
        )

        # train model
        trainer.fit(train_loader,
                    val_loader,
                    max_epochs=50,
                    default_root_dir=f'./.tmp_out/examples_pipeline/model/{i}',
                    callbacks=[early_stopping,checkpoint_callback],
                    enable_progress_bar=True,
                    enable_model_summary=True,
                    limit_val_batches=1.0)
        
        # test model
        score = trainer.test(val_loader,
                            enable_progress_bar=True,
                            enable_model_summary=True)[0]
        print(f'Fold {i} test accuracy: {score["test_accuracy"]:.4f}')