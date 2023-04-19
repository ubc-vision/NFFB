import torch
from torch import nn
import numpy as np


class Sine(nn.Module):
    def __init__(self, w0):
        super().__init__()

        self.w0 = w0

    def forward(self, input):
        return torch.sin(input * self.w0)



def sine_init(m, w0, num_input=None):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            if num_input is None:
                num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1.0 / num_input, 1.0 / num_input)
