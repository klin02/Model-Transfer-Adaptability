from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import os.path as osp
import time
# import sys

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.
    lossLayer = nn.CrossEntropyLoss()
    start_time = time.time()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data,targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, targets)
        loss.backward()
        total_loss += loss.item() * len(data)
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)

        if batch_idx % 200 == 0 and batch_idx > 0:
            cur_loss = total_loss / 200
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                'loss {:5.2f}'.format(
                epoch, batch_idx, len(train_loader.dataset) // len(data), lr,
                elapsed * 1000 / 200, cur_loss))
            
            total_loss = 0.
            correct = 0
            start_time = time.time()


def evaluate(model, device, eval_loader):
    model.eval()
    total_loss = 0
    correct = 0
    lossLayer = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, targets in eval_loader:
            data,targets = data.to(device), targets.to(device)
            output = model(data)
            total_loss += len(data) * lossLayer(output, targets).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
    
    test_loss = total_loss / len(eval_loader.dataset)
    test_acc = 100. * correct / len(eval_loader.dataset)
    return test_loss,test_acc


epochs_cfg_table = {
    'AlexNet'       : [20, 30, 20, 20, 10],
    'AlexNet_BN'    : [15, 20, 20, 20, 10, 10],
    'VGG_16'        : [25, 30, 30, 20, 20, 10, 10],  
    'VGG_19'        : [30, 40, 30, 20, 20, 10, 10],
    'Inception_BN'  : [20, 30, 30, 20, 20, 10, 10],
    'ResNet_18'     : [30, 25, 25, 20, 10, 10],
    'ResNet_50'     : [30, 40, 35, 25, 15, 10, 10],
    'ResNet_152'    : [50, 60, 50, 40, 25, 15, 10, 10],
    'MobileNetV2'   : [25, 35, 30, 20, 10, 10], 
}
lr_cfg_table = {
    'AlexNet'       : [0.01, 0.005, 0.001, 0.0005, 0.0001],
    'AlexNet_BN'    : [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001],
    'VGG_16'        : [0.01, 0.008, 0.005, 0.002, 0.001, 0.0005, 0.0001],  
    'VGG_19'        : [0.01, 0.008, 0.005, 0.002, 0.001, 0.0005, 0.0001],
    'Inception_BN'  : [0.01, 0.008, 0.005, 0.002, 0.001, 0.0005, 0.0001],
    'ResNet_18'     : [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001],
    'ResNet_50'     : [0.01, 0.008, 0.005, 0.002, 0.001, 0.0005, 0.0001],
    'ResNet_152'    : [0.01, 0.008, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0001],
    'MobileNetV2'   : [0.01, 0.008, 0.005, 0.002, 0.001, 0.0001],
}
if __name__ == "__main__":
    # sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

    batch_size = 32
    seed = 1111
    seed_gpu = 1111
    lr = 0.05   # origin lr
    # momentum = 0.5
    t_epochs = 300 #学习率衰减周期
    patience = 30 #早停参数
    save_model = True

    append = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed_gpu)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32), interpolation=InterpolationMode.BICUBIC),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    if save_model:
        if not osp.exists('ckpt'):
            os.makedirs('ckpt')

    # model_name_list = ['AlexNet', 'AlexNet_BN', 'VGG_16', 'VGG_19', 'Inception_BN', 
    #                    'ResNet_18', 'ResNet_50', 'ResNet_152', 'MobileNetV2']
    # model_name_list = ['ResNet_18', 'ResNet_50', 'ResNet_152', 'MobileNetV2']
    model_name_list = ['ResNet_152']
    for model_name in model_name_list:
        save_path = 'ckpt/cifar10_'+model_name+'.pt'
        if os.path.exists(save_path) and append:
            continue
        else:
            print('>>>>>>>>>>>>>>>>>>>>>>>> Train: '+model_name+' <<<<<<<<<<<<<<<<<<<<<<<<')
            model = Model(model_name).to(device)

            best_val_acc = None
            optimizer = optim.SGD(model.parameters(), lr=lr)
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=t_epochs)
            weak_cnt = 0 # 弱于最佳精度的计数器
            epoch = 0
            while weak_cnt < patience:
                epoch += 1
                epoch_start_time = time.time()
                train(model, device, train_loader, optimizer, epoch)
                val_loss, val_acc = evaluate(model, device, test_loader)
                if not best_val_acc or val_acc > best_val_acc:
                    best_val_acc = val_acc
                    weak_cnt = 0
                    if save_model:
                        torch.save(model.state_dict(), save_path)
                else:
                    weak_cnt += 1
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                        'test acc {:.2f} | weak_cnt {:d}'.format(epoch, (time.time() - epoch_start_time),
                                                val_loss, val_acc, weak_cnt))
                print('-' * 89)
                lr_scheduler.step()

            print('>>> Early Stop: No improvement after patience(%d) epochs.'%patience)
        
        model = Model(model_name).to(device)
        model.load_state_dict(torch.load(save_path))

        test_loss,test_acc = evaluate(model, device, test_loader)
        print('=' * 89)
        print('| Test on {:s} | test loss {:5.2f} | test acc {:.2f}'.format(
            model_name, test_loss, test_acc))
        print('=' * 89)