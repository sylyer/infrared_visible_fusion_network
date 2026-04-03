import torch
import torch.nn as nn

class DepthwiseCNN(nn.Module):
    """Lightweight Depthwise Separable CNN backbone for feature extraction"""
    def __init__(self, input_channels=1, num_classes=None):
        super(DepthwiseCNN, self).__init__()
        self.input_channels = input_channels
        
        # Depthwise Separable Conv Layer 1
        self.conv1_dw = nn.Conv2d(input_channels, input_channels, kernel_size=3, 
                                   stride=1, padding=1, groups=input_channels, bias=False)
        self.conv1_pw = nn.Conv2d(input_channels, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Depthwise Separable Conv Layer 2
        self.conv2_dw = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.conv2_pw = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Depthwise Separable Conv Layer 3
        self.conv3_dw = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.conv3_pw = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Layer 1
        x = self.conv1_dw(x)
        x = self.conv1_pw(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2_dw(x)
        x = self.conv2_pw(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Layer 3
        x = self.conv3_dw(x)
        x = self.conv3_pw(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        return x