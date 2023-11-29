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
from analyse_utils import *


def direct_quantize(model, test_loader,device):
    for i, (data, target) in enumerate(test_loader, 1):
        data = data.to(device)
        output = model.quantize_forward(data).cpu()
        if i % 100 == 0:
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
    batch_size = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=False,
                     transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=2),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
     batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, download=False, 
                     transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                (0.2023, 0.1994, 0.2010))
                            ])),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    load_ptq = True
    store_ptq = True

    append = True
    gol._init()

    # ckpt_path = 'ckpt_sgd_5_momentum_95'
    ckpt_path = 'ckpt_adam_1'
    
    model_name_list = ['AlexNet','AlexNet_BN','VGG_16','VGG_19','Inception_BN',
                       'MobileNetV2','ResNet_18','ResNet_50','ResNet_152']
    # model_name_list = ['AlexNet','AlexNet_BN','VGG_16','VGG_19','Inception_BN',
    #                    'MobileNetV2','ResNet_18','ResNet_50']
    # model_name_list = ['MobileNetV2']
    model_name = sys.argv[1]

    model = Model(model_name)
    full_file = ckpt_path+'/cifar10_'+model_name+'.pt'
    model.load_state_dict(torch.load(full_file))
    model.to(device)

    ptq_path = ckpt_path+'/'+model_name
    if not osp.exists(ptq_path):
        os.makedirs(ptq_path)
    ptq_file_prefix = ptq_path+'/cifar10_'+model_name+'_ptq_'

    model.eval()
    full_acc = full_inference(model, test_loader, device)
    # 传入后可变
    fold_model(model)

    Mac,Param,layer, par_ratio, flop_ratio = extract_ratio(model_name)
    par_ratio, flop_ratio = fold_ratio(layer, par_ratio, flop_ratio)

    full_names = []
    full_params = []

    writer = SummaryWriter(log_dir='./log/'+model_name)
    for name, param in model.named_parameters():
        if 'conv' in name or 'fc' in name:
            full_names.append(name)
            full_params.append(param.data.cpu())
            writer.add_histogram(tag='Full_' + name, values=param.data)

    
    quant_type_list = ['INT','POT','FLOAT']
    # quant_type_list = ['INT']
    # quant_type_list = ['POT']
    # quant_type_list = ['INT','POT']
    title_list = []

    for quant_type in quant_type_list:
        num_bit_list = numbit_list(quant_type)
        # 对一个量化类别，只需设置一次bias量化表
        # int由于位宽大，使用量化表开销过大，直接_round即可
        if quant_type != 'INT':
            bias_list = build_bias_list(quant_type)
            gol.set_value(bias_list, is_bias=True)

        for num_bits in num_bit_list:
            e_bit_list = ebit_list(quant_type,num_bits)
            for e_bits in e_bit_list:
                model_ptq = Model(model_name)
                if quant_type == 'FLOAT':
                    title = '%s_%d_E%d' % (quant_type, num_bits, e_bits)
                else:
                    title = '%s_%d' % (quant_type, num_bits)
                print('\n'+model_name+': PTQ: '+title)
                title_list.append(title)

                # 设置量化表
                if quant_type != 'INT':
                    plist = build_list(quant_type, num_bits, e_bits)
                    gol.set_value(plist)
                # 判断是否需要载入
                if load_ptq is True and osp.exists(ptq_file_prefix + title + '.pt'):
                    model_ptq.quantize(quant_type,num_bits,e_bits)
                    model_ptq.load_state_dict(torch.load(ptq_file_prefix + title + '.pt'))
                    model_ptq.to(device)
                    print('Successfully load ptq model: ' + title)
                else:
                    model_ptq.load_state_dict(torch.load(full_file))
                    model_ptq.to(device)
                    model_ptq.quantize(quant_type,num_bits,e_bits)
                    model_ptq.eval()
                    direct_quantize(model_ptq, train_loader, device)
                    if store_ptq:
                        torch.save(model_ptq.state_dict(), ptq_file_prefix + title + '.pt')

                model_ptq.freeze()
                ptq_acc = quantize_inference(model_ptq, test_loader, device)
                acc_loss = (full_acc - ptq_acc) / full_acc

                #将量化后分布反量化到全精度相同的scale
                model_ptq.fakefreeze()
                # 获取计算量/参数量下的js-div
                js_flops = 0.
                js_param = 0.
                for name, param in model_ptq.named_parameters():
                    if 'conv' not in name and 'fc' not in name:
                        continue
                    writer.add_histogram(tag=title+'_'+name,values=param.data)
                    print('-'*89)
                    print(name)
                    prefix = name.rsplit('.',1)[0]
                    layer_idx = layer.index(prefix)
                    name_idx = full_names.index(name)

                    layer_idx = layer.index(prefix)
                    ptq_param = param.data.cpu()

                    js = js_div(ptq_param,full_params[name_idx])

                    js = js.item()
                    if js < 0.:
                        js = 0.
                    js_flops_layer = js * flop_ratio[layer_idx]
                    js_param_layer = js * par_ratio[layer_idx]
                    print('js_flops:%f'%js_flops_layer)
                    print('js_param:%f'%js_param_layer)
                    
                    if quant_type == 'INT':
                        if 'bias' in name:
                            plist = build_int_list(16)
                        else:
                            plist = build_int_list(num_bits)
                    else:
                        if 'bias' in name:
                            plist = gol.get_value(True)
                        else:
                            plist = gol.get_value(False)
                    list_std_mid_ratio = std_mid_ratio(plist)
                    list_range_mid_ratio = range_mid_ratio(plist)
                    full_std_mid_ratio = std_mid_ratio(full_params[name_idx])
                    full_range_mid_ratio = range_mid_ratio(full_params[name_idx])
                    ptq_std_mid_ratio = std_mid_ratio(ptq_param)
                    ptq_range_mid_ratio = range_mid_ratio(ptq_param)
                    print('std_mid_ratio:')
                    print('  plist:%f'%list_std_mid_ratio)
                    print('  full :%f'%full_std_mid_ratio)
                    print('  ptq  :%f'%ptq_std_mid_ratio)
                    print('range_mid_ratio:')
                    print('  plist:%f'%list_range_mid_ratio)
                    print('  full :%f'%full_range_mid_ratio)
                    print('  ptq  :%f'%ptq_range_mid_ratio)
                    cossim = cos_sim(full_params[name_idx],ptq_param)
                    print('cossim:%f'%cossim)
                    pearson = pearson_corr(full_params[name_idx],ptq_param)
                    print('pearson:%f'%pearson)
                    list_kurt = kurtosis(plist)
                    full_kurt = kurtosis(full_params[name_idx])
                    ptq_kurt = kurtosis(ptq_param)
                    print('kurtosis:')
                    print('  plist:%f'%list_kurt)
                    print('  full :%f'%full_kurt)
                    print('  ptq  :%f'%ptq_kurt)
                    js_flops = js_flops + js * flop_ratio[layer_idx]
                    js_param = js_param + js * par_ratio[layer_idx]

                print(title + ': js_flops: %f js_param: %f acc_loss: %f' % (js_flops,js_param, acc_loss))
                print('='*89)
        
    writer.close()


