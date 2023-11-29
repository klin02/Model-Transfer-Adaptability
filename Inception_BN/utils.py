import torch
import torch.nn as nn

def ebit_list(quant_type, num_bits):
    if quant_type == 'FLOAT':
        e_bit_list = list(range(1,num_bits-1))
    else:
        e_bit_list = [0]
    return e_bit_list


def numbit_list(quant_type):
    if quant_type == 'INT':
        num_bit_list = list(range(2,17))
        # num_bit_list = [4,5]
    elif quant_type == 'POT':
        num_bit_list = list(range(2,9))
        # num_bit_list = [5]
    else:
        num_bit_list = list(range(2,9))
        # num_bit_list = [8]
    
    return num_bit_list     

def build_bias_list(quant_type):
    if quant_type == 'POT':
        return build_pot_list(8)
    else:
        return build_float_list(16,7)
    
def build_list(quant_type, num_bits, e_bits):
    if quant_type == 'POT':
        return build_pot_list(num_bits)
    else:
        return build_float_list(num_bits,e_bits)

def build_pot_list(num_bits):
    plist = [0.]
    for i in range(-2 ** (num_bits-1) + 2, 1): 
        # i最高到0，即pot量化最大值为1
        plist.append(2. ** i)
        plist.append(-2. ** i)
    plist = torch.Tensor(list(set(plist)))
    # plist = plist.mul(1.0 / torch.max(plist))
    return plist

def build_float_list(num_bits,e_bits):
    m_bits = num_bits - 1 - e_bits
    plist = [0.]
    # 相邻尾数的差值
    dist_m = 2 ** (-m_bits)
    e = -2 ** (e_bits - 1) + 1
    for m in range(1, 2 ** m_bits):
        frac = m * dist_m   # 尾数部分
        expo = 2 ** e       # 指数部分
        flt = frac * expo
        plist.append(flt)
        plist.append(-flt)

    for e in range(-2 ** (e_bits - 1) + 2, 2 ** (e_bits - 1) + 1):
        expo = 2 ** e
        for m in range(0, 2 ** m_bits):
            frac = 1. + m * dist_m
            flt = frac * expo
            plist.append(flt)
            plist.append(-flt)
    plist = torch.Tensor(list(set(plist)))
    return plist

#此处不必cfg，直接取同前缀同后缀即可。将relu一起考虑进去
def fold_ratio(layer, par_ratio, flop_ratio):
    idx = -1
    for name in layer:
        if 'conv' in name:
            conv_idx = layer.index(name)
            [prefix,suffix] = name.split('conv')
            bn_name = prefix+'bn'+suffix
            relu_name = prefix+'relu'+suffix
            if bn_name in layer:
                bn_idx = layer.index(bn_name)
                par_ratio[conv_idx]+=par_ratio[bn_idx]
                flop_ratio[conv_idx]+=flop_ratio[bn_idx]
                if relu_name in layer:
                    relu_idx = layer.index(relu_name)
                    par_ratio[conv_idx]+=par_ratio[relu_idx]
                    flop_ratio[conv_idx]+=flop_ratio[bn_idx]
    return par_ratio,flop_ratio

def fold_model(model):
    for name, module in model.named_modules():
        if 'conv' in name:
            [prefix,suffix] = name.split('conv')
            bn_name = prefix+'bn'+suffix
            if hasattr(model,bn_name):
                bn_layer = getattr(model,bn_name)
                fold_bn(module,bn_layer)

def fold_bn(conv, bn):
    # 获取 BN 层的参数
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = torch.sqrt(var + eps)

    if bn.affine:
        gamma_ = bn.weight / std
        weight = conv.weight * gamma_.view(conv.out_channels, 1, 1, 1)
        if conv.bias is not None:
            bias = gamma_ * conv.bias - gamma_ * mean + bn.bias
        else:
            bias = bn.bias - gamma_ * mean
    else:
        gamma_ = 1 / std
        weight = conv.weight * gamma_
        if conv.bias is not None:
            bias = gamma_ * conv.bias - gamma_ * mean
        else:
            bias = -gamma_ * mean

    # 设置新的 weight 和 bias
    conv.weight.data = weight.data
    if conv.bias is not None:
        conv.bias.data = bias.data
    else:
        conv.bias = torch.nn.Parameter(bias)
