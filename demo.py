import torch
import numpy as np

data = [[1],[3]]
x_data = torch.tensor(data, device='cuda', dtype=torch.float)

# print(x_data)
print(x_data)

weights = [[2],[4]]
bias = 7

weights_tensor = torch.tensor(weights, device='cuda', dtype=torch.float)
bias_tensor = torch.tensor([bias], device='cuda', dtype=torch.float)

# one neuron

print(weights_tensor.T.shape)
# print(bias_tensor.shape)

output = weights_tensor.T @ x_data + bias_tensor

print(output)
print(output.shape)
print(output.device)

