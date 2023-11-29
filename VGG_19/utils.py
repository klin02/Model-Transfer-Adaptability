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
    elif quant_type == 'POT':
        num_bit_list = list(range(2,9))
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

def fold_ratio(layer, par_ratio, flop_ratio):
    idx = -1
    for name in layer:
        idx = idx + 1
        if 'bn' in name:
            par_ratio[idx-1] += par_ratio[idx]
            flop_ratio[idx-1] += flop_ratio[idx]
    return par_ratio,flop_ratio

def fold_model(model):
    idx = -1
    module_list = []
    for name, module in model.named_modules():
        idx += 1
        module_list.append(module)
        if 'bn' in name:
            module_list[idx-1] = fold_bn(module_list[idx-1],module)
    return model
# def fold_model(model):
#     last_conv = None
#     last_bn = None

#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):
#             # 如果当前模块是卷积层，则将其 "fold" 到上一个 BN 层中
#             if last_bn is not None:
#                 last_conv = fold_bn(last_conv, last_bn)
#                 last_bn = None
#             last_conv = module

#         elif isinstance(module, nn.BatchNorm2d):
#             # 如果当前模块是 BN 层，则将其 "fold" 到上一个卷积层中
#             last_bn = module
#             if last_conv is not None:
#                 last_conv = fold_bn(last_conv, last_bn)
#                 last_bn = None

#     # 处理最后一个 BN 层
#     if last_bn is not None:
#         last_conv = fold_bn(last_conv, last_bn)

#     return model

def fold_bn(conv, bn):
    # 获取 BN 层的参数
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = torch.sqrt(var + eps)
    feat = bn.num_features
    
    # 获取卷积层的参数
    weight = conv.weight.data
    bias = conv.bias.data

    if bn.affine:
        gamma_ = gamma / std
        weight = weight * gamma_.view(feat, 1, 1, 1)
        if bias is not None:
            bias = gamma_ * bias - gamma_ * mean + beta
        else:
            bias = beta - gamma_ * mean
    else:
        gamma_ = 1 / std
        weight = weight * gamma_
        if bias is not None:
            bias = gamma_ * bias - gamma_ * mean
        else:
            bias = -gamma_ * mean

    # 设置新的 weight 和 bias
    conv.weight.data = weight
    conv.bias.data = bias

    return conv