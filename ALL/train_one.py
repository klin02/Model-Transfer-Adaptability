from model import *
from dataloader import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import _LRScheduler,MultiStepLR
import os
import os.path as osp
import time
import sys

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.
    lossLayer = nn.CrossEntropyLoss()
    #对第一个epoch，使用warmup策略
    if epoch == 1:
        warmup_scheduler = WarmUpLR(optimizer, len(train_loader))
    else:
        warmup_scheduler = None

    start_time = time.time()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data,targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, targets)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

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
    model_name = sys.argv[1]
    dataset = sys.argv[2]

    batch_size = 128
    optim_type = 'sgd' 
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    nesterov = True
    
    epochs = 200
    milestones = [60, 120, 160]
    gamma = 0.2

    print('model: '+model_name+'  dataset: '+dataset)
    print('optim_type: '+optim_type+'  lr: '+str(lr)+'  weight_decay: '+str(weight_decay)+'  nesterov: '+str(nesterov)+'  momentum: '+str(momentum))
    print('epochs: '+str(epochs)+'  milestones: '+str(milestones)+'  gamma: '+str(gamma))

    save_model = True
    append = False

    ckpt_path = 'ckpt_full/'+dataset
    if save_model:
        if not osp.exists(ckpt_path):
            os.makedirs(ckpt_path)
    save_path = ckpt_path+'/'+model_name+'.pt'
    if os.path.exists(save_path) and append:
        print('Append: Model '+model_name+' exists!')
        sys.exit()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataloader = DataLoader(dataset,batch_size)
    train_loader, val_loader, test_loader = dataloader.getloader()
    

    print('>>>>>>>>>>>>>>>>>>>>>>>> Train: '+model_name+' <<<<<<<<<<<<<<<<<<<<<<<<')
    model = Model(model_name,dataset).to(device)

    best_val_acc = None

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay = weight_decay,nesterov=nesterov)

    lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, device, train_loader, optimizer, epoch)
        val_loss, val_acc = evaluate(model, device, val_loader)
        if not best_val_acc or val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_model:
                torch.save(model.state_dict(), save_path)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.2f} | '
                'val acc {:.2f} | best val acc {:.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, val_acc, best_val_acc))
        print('-' * 89)
        lr_scheduler.step(epoch)

    print('>>>>>>>>>>>>>>>>>>>>>>>> Test: '+model_name+' <<<<<<<<<<<<<<<<<<<<<<<<')        
    model = Model(model_name,dataset).to(device)
    model.load_state_dict(torch.load(save_path))

    test_loss,test_acc = evaluate(model, device, test_loader)
    print('=' * 89)
    print('| Test on {:s} | test loss {:5.2f} | test acc {:.2f}'.format(
        model_name, test_loss, test_acc))
    print('=' * 89)