from model import *

import torch
from ptflops import get_model_complexity_info


if __name__ == "__main__":
    model = AlexNet()
    full_file = 'ckpt/cifar10_AlexNet.pt'
    model.load_state_dict(torch.load(full_file))

    flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)

