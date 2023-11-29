from torch.serialization import load
from model import *
from extract_ratio import *
from utils import *

import gol
import openpyxl
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import torch.utils.bottleneck as bn
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter


def direct_quantize(model, test_loader,device):
    for i, (data, target) in enumerate(test_loader, 1):
        data = data.to(device)
        output = model.quantize_forward(data).cpu()
        if i % 500 == 0:
            break
    print('direct quantization finish')


def full_inference(model, test_loader, device):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data = data.to(device)
        output = model(data).cpu()
        pred = output.argmax(dim=1, keepdim=True)
        # print(pred)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def quantize_inference(model, test_loader, device):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        data = data.to(device)
        output = model.quantize_inference(data).cpu()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('Test set: Quant Model Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


if __name__ == "__main__":
    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = Inception_BN()
    writer = SummaryWriter(log_dir='./log')
    full_file = 'ckpt/cifar10_Inception_BN.pt'
    model.load_state_dict(torch.load(full_file))
    model.to(device)

    load_ptq = True
    store_ptq = False
    ptq_file_prefix = 'ckpt/cifar10_Inception_BN_ptq_'

    model.eval()
    # 传入后可变
    fold_model(model)
    
    layer, par_ratio, flop_ratio = extract_ratio()
    par_ratio, flop_ratio = fold_ratio(layer, par_ratio, flop_ratio)

    full_names = []
    full_params = []
    full_mid_ratios = []
    full_params_norm = []

    for name, param in model.named_parameters():
        if 'conv' in name or 'fc' in name:
            full_names.append(name)
            param_norm = F.normalize(param.data.cpu(),p=2,dim=-1)
            full_params.append(param.data.cpu())
            full_mr = mid_ratio(param.data.cpu())
            full_mid_ratios.append(full_mr)
            print(name+':%f'%full_mr)
            full_params_norm.append(param_norm)
            writer.add_histogram(tag='Full_' + name + '_data', values=param.data)