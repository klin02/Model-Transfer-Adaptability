import torch
from utils import *
from module import *
from torch.utils.tensorboard import SummaryWriter

def build_int_list(num_bits):
    plist = [0.]
    for i in range(0,2**(num_bits-1)): 
        # i最高到0，即pot量化最大值为1
        plist.append(i)
        plist.append(i)
    plist = torch.Tensor(list(set(plist)))
    # plist = plist.mul(1.0 / torch.max(plist))
    return plist


if __name__ == "__main__":
    writer = SummaryWriter(log_dir='./log')
    quant_type_list = ['INT','POT','FLOAT']
    for quant_type in quant_type_list:
        num_bit_list = numbit_list(quant_type)
        for num_bits in num_bit_list:
            e_bit_list = ebit_list(quant_type,num_bits)
            for e_bits in e_bit_list:
                if quant_type == 'FLOAT':
                    title = '%s_%d_E%d' % (quant_type, num_bits, e_bits)
                else:
                    title = '%s_%d' % (quant_type, num_bits)
                print('\nPTQ: '+title)
                # 设置量化表
                if quant_type != 'INT':
                    plist = build_list(quant_type, num_bits, e_bits)
                    list_mid_ratio = mid_ratio(plist)
                else:
                    plist = build_int_list(num_bits)
                    list_mid_ratio = mid_ratio(plist)
                writer.add_histogram(tag=title, values=plist)
                print(list_mid_ratio)
                
    writer.close()