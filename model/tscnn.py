#------------------------ Header Comment ----------------------#
# File: tscnn.py
# Date: 2024-11-03
# Author: Weizhi Peng
#--------------------------------------------------------------#
# Purpose:
# This script defines a Temporal-Spatial Convolutional Neural Network 
# (TSCNN) for EEG-based classification tasks. The network integrates 
# a lightweight spatial attention module to enhance feature extraction 
# and improve the classification performance. The TSCNN model is designed 
# to take grid-structured EEG data as input and output classification 
# results for multiple classes.
#
# Main Components:
# 1. SpatialAttentionModule: A module that applies spatial attention 
#    to the input feature map to emphasize significant spatial information.
# 2. TSCNN Model: A neural network composed of multiple convolutional 
#    layers, batch normalization, ReLU activation, and a fully connected 
#    layer for classification.
#
# Attributes of TSCNN:
# - Conv1, Conv2, Conv3, Conv4: Convolutional layers for feature extraction.
# - Spatial Attention: Enhances the features after the third convolutional block.
# - Linear Layers: Fully connected layers for final classification.
#
# Requirements:
# - Python (3.7+)
# - PyTorch
#
# Notes:
# - The spatial attention module helps the network focus on relevant spatial 
#   features in the input EEG data.
# - The TSCNN model is flexible for use in multi-class classification tasks.
#--------------------------------------------------------------#



from typing import Tuple

import torch
import torch.nn as nn

class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module that applies spatial attention to the input feature map.

    Args:
        kernel_size (int): Size of the convolutional kernel. Default is 7.

    Methods:
        forward(x):
            Forward pass of the module.
            
            Args:
                x (torch.Tensor): Input feature map of shape (batch_size, channels, height, width).
            
            Returns:
                torch.Tensor: Output feature map after applying spatial attention.
    """
    def __init__(self, kernel_size=7):
        """
        Initializes the SpatialAttentionModule.

        Args:
            kernel_size (int, optional): The size of the convolutional kernel. Default is 7.
        """
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W) where
                              N is the batch size,
                              C is the number of channels,
                              H is the height, and
                              W is the width.

        Returns:
            torch.Tensor: Output tensor after applying the forward pass operations.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

# Integrating SpatialAttentionModule Block into CCNN
class TSCNN(nn.Module):
    """
    Temporal-Spatial Convolutional Neural Network (TSCNN) for EEG classification.
    Args:
        in_channels (int): Number of input channels. Default is 4.
        grid_size (Tuple[int, int]): Size of the input grid. Default is (9, 9).
        num_classes (int): Number of output classes. Default is 2.
        dropout (float): Dropout rate for the dropout layer. Default is 0.5.
    Attributes:
        in_channels (int): Number of input channels.
        grid_size (Tuple[int, int]): Size of the input grid.
        num_classes (int): Number of output classes.
        dropout (float): Dropout rate for the dropout layer.
        conv1 (nn.Sequential): First convolutional layer with padding, Conv2d, BatchNorm2d, and ReLU.
        conv2 (nn.Sequential): Second convolutional layer with padding, Conv2d, BatchNorm2d, and ReLU.
        conv3 (nn.Sequential): Third convolutional layer with padding, Conv2d, BatchNorm2d, ReLU, and SpatialAttentionModule.
        conv4 (nn.Sequential): Fourth convolutional layer with padding, Conv2d, BatchNorm2d, and ReLU.
        lin1 (nn.Sequential): First fully connected layer with Linear, SELU, and Dropout2d.
        lin2 (nn.Linear): Second fully connected layer with Linear.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass of the network.
            Args:
                x (torch.Tensor): Input t
                
                
                ensor.
            Returns:
                torch.Tensor: Output tensor after passing through the network.
    """
    def __init__(self, in_channels: int = 4, grid_size: Tuple[int, int] = (9, 9), num_classes: int = 2, dropout: float = 0.5):
        """
        Initializes the TSCNN model.
        Args:
            in_channels (int): Number of input channels. Default is 4.
            grid_size (Tuple[int, int]): Size of the input grid. Default is (9, 9).
            num_classes (int): Number of output classes. Default is 2.
            dropout (float): Dropout rate for the dropout layer. Default is 0.5.
        """
        super(TSCNN, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)), 
            nn.Conv2d(self.in_channels, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)), 
            nn.Conv2d(64, 128, kernel_size=4, stride=1), 
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)), 
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SpatialAttentionModule()  # Lightweight spatial attention
        )
        self.conv4 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)), 
            nn.Conv2d(256, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.lin1 = nn.Sequential(
            nn.Linear(self.grid_size[0] * self.grid_size[1] * 64, 1024),
            nn.SELU(),
            nn.Dropout2d(self.dropout)
        )
        self.lin2 = nn.Linear(1024, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network layers.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x
