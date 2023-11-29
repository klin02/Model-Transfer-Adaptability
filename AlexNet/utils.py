import torch

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