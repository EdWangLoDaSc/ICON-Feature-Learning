import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import tqdm.notebook as tqdm

import random
import time

# from google.colab import drive
from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import plotly.io as pio
if IN_COLAB:
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "vscode"

import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
        self.ctx = None
    
    def add_hook(self, hook, dir='fwd'):
        if dir == 'fwd':
            self.fwd_hooks.append(hook)
        elif dir == 'bwd':
            self.bwd_hooks.append(hook)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if dir == 'fwd':
            self.fwd_hooks = []
        elif dir == 'bwd':
            self.bwd_hooks = []
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        self.ctx = x
        for hook in self.fwd_hooks:
            x = hook(x)
        return x
