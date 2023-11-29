from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import os.path as osp
import time
import sys

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
        total_loss += loss.item()
        optimizer.step()

        if batch_idx % 50 == 0 and batch_idx > 0:
            cur_loss = total_loss / 50
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
                'loss {:5.2f}'.format(
                epoch, batch_idx, len(train_loader.dataset) // len(data), lr,
                elapsed * 1000 / 50, cur_loss))
            
            total_loss = 0.
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

if __name__ == "__main__":
    # sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

    batch_size = 128
    optim_type = 'sgd' 
    # optim_type = 'sgd'
    if optim_type is 'adam':
        lr = 0.0001
        opt_path = 'adam_lr'+str(lr).split('.')[-1]
    elif optim_type is 'sgd':
        lr = 0.01
        momentum = 0.9
        weight_decay = 1e-4
        nesterov = True
        opt_path = 'sgd_lr'+str(lr).split('.')[-1]+'_wd'+str(weight_decay).split('.')[-1]+'_ntr'

    # lr = 0.001   # origin lr
    # end_lr_bound = 0.00001 #早停时学习率应当低于该值，从而保证充分收敛
    # momentum = 0.9
    # weight_decay = 1e-4
    # nesterov = True
    t_epochs = 800 #学习率衰减周期
    patience = 50 #早停参数
    
    print(opt_path)
    print('t_epoches:%d patience:%d'%(t_epochs,patience))
    save_model = True

    append = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transfrom = transforms.Compose([
                        transforms.RandomCrop(32, padding=2),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
    eval_transfrom = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010))
                        ])

    alltrainset = datasets.CIFAR100(root='/lustre/datasets/CIFAR100',train=True,download=True,transform=train_transfrom)
    
    train_size = (int)(0.8 * len(alltrainset))
    val_size = (int)(0.2 * len(alltrainset))
    
    train_idx, val_idx = torch.utils.data.random_split(range(train_size+val_size),[train_size,val_size])

    trainset = torch.utils.data.Subset(alltrainset,train_idx)
    valset = torch.utils.data.Subset(alltrainset,val_idx)

    train_loader = torch.utils.data.DataLoader(
     trainset,
     batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('/lustre/datasets/CIFAR100', train=False, download=False, 
                     transform=eval_transfrom),
    batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # ckpt_path = 'ckpt_sgd_1_momentum_9_ntb'
    ckpt_path = 'ckpt_'+opt_path
    if save_model:
        if not osp.exists(ckpt_path):
            os.makedirs(ckpt_path)

    model_name = sys.argv[1]
    save_path = ckpt_path+'/cifar100_'+model_name+'.pt'
    if os.path.exists(save_path) and append:
        pass
    else:
        print('>>>>>>>>>>>>>>>>>>>>>>>> Train: '+model_name+' <<<<<<<<<<<<<<<<<<<<<<<<')
        model = Model(model_name).to(device)

        best_val_acc = None
        if 'adam' in opt_path:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif 'sgd' in opt_path:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay = weight_decay,nesterov=nesterov)
        else:
            raise ValueError('Illegal opttype')
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=t_epochs)
        weak_cnt = 0 # 弱于最佳精度的计数器
        epoch = 0
        while True:
            epoch += 1
            epoch_start_time = time.time()
            train(model, device, train_loader, optimizer, epoch)
            val_loss, val_acc = evaluate(model, device, val_loader)
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
            
            #这里额外限制早停时，学习率曾低于一个限度，保证充分训练
            if weak_cnt >= patience:
                break
                # if optimizer.param_groups[0]['lr'] < end_lr_bound:
                #     break
                # elif epoch > t_epochs: # 已经训练完一个周期，曾达到最小精度
                #     break
                # else: #若多个epoch未提升，载入当前最佳精度模型继续训练
                #     print('>> Turn Back: No improvement after %d epochs, back to BestPoint'%patience)
                #     weak_cnt = 0
                #     model.load_state_dict(torch.load(save_path))

        print('>>> Early Stop: No improvement after patience(%d) epochs'%patience)
        # print('>>> Early Stop: No improvement after patience(%d) epochs And lr under bound(%f) '%(patience,end_lr_bound))
    
    model = Model(model_name).to(device)
    model.load_state_dict(torch.load(save_path))

    test_loss,test_acc = evaluate(model, device, test_loader)
    print('=' * 89)
    print('| Test on {:s} | test loss {:5.2f} | test acc {:.2f}'.format(
        model_name, test_loss, test_acc))
    print('=' * 89)