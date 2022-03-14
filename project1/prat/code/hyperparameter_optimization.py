import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms 
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from models import ResNet, BasicBlock



B_i = {
    'B_1': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'B_2': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'B_3': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

def get_model_params(B_i):

    #print('N:', print(N))
    hyper_params = []
    for i in B_i['B_1']:
        for j in B_i['B_2']:
            for k in B_i['B_3']:
                model = ResNet(BasicBlock, [i, j, k])
                
                param_num = sum(p.numel() for p in model.parameters())
                if (param_num < 5000000) and (param_num > 4800000):
                    hyper_params.append([i, j, k])
                    print(param_num)

    return hyper_params
                    


hyper_params = get_model_params(B_i)
print(len(hyper_params))
print(hyper_params)