# Simple Inference

# Run inference on a PyTorch model. Given an input tensor and a trained torch.nn.Linear model, compute the forward pass and store the result in the output tensor.

# The model performs a linear transformation: output = input @ weight.T + bias where weight has shape [output_size, input_size] and bias has shape [output_size].

# Constraints
# 1 ≤ batch_size ≤ 1,000
# 1 ≤ input_size ≤ 1,000
# 1 ≤ output_size ≤ 1,000
# -10.0 ≤ input values ≤ 10.0

import torch
import torch.nn as nn


# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    with torch.no_grad():
        output.copy_(model(input))
        
