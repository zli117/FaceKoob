"""Define the embedding model
"""
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Module, functional


class Embedder(Module):
    def __init__(self, input_size, kernel_sizes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_sizes[0])
        self.pool1 = nn.MaxPool2d(kernel_size=kernel_sizes[1], stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_sizes[2])
        self.pool2 = nn.MaxPool2d(kernel_size=kernel_sizes[3], stride=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=kernel_sizes[4])
        self.pool3 = nn.AvgPool2d(kernel_size=kernel_sizes[5], stride=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=kernel_sizes[6])
        self.pool4 = nn.AvgPool2d(kernel_size=kernel_sizes[7], stride=1)
        size_reduction = sum(kernel_sizes) - len(kernel_sizes)
        self.fc_input_dimension = ((input_size[0] - size_reduction) *
                                   (input_size[1] - size_reduction) * 32)
        self.fc1 = nn.Linear(self.fc_input_dimension, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = Variable(x, requires_grad=False)
        x = functional.relu(self.pool1(self.conv1(x)))
        x = functional.dropout2d(x, p=0.5)
        x = functional.relu(self.pool2(self.conv2(x)))
        x = functional.dropout2d(x, p=0.2)
        x = functional.relu(self.pool3(self.conv3(x)))
        x = functional.dropout2d(x, p=0.1)
        x = functional.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, self.fc_input_dimension)
        x = functional.relu(self.fc1(x))
        x = functional.sigmoid(self.fc2(x))
        return x
