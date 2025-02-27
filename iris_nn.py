import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            # input layer
            nn.Linear(4, 1000),
            nn.ReLU(),
            # hidden layer 1
            nn.Linear(1000, 500),
            nn.ReLU(),
            # hidden layer 2
            nn.Linear(500, 300),
            nn.ReLU(),
            # output layer
            nn.Linear(300, 3)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits