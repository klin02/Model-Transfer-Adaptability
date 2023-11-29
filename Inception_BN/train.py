from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import os
import os.path as osp


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))


if __name__ == "__main__":
    batch_size = 32
    seed = 1
    epochs_cfg = [20, 30, 30, 20, 20, 10, 10]
    lr_cfg = [0.01, 0.008, 0.005, 0.002, 0.001, 0.0005, 0.0001]
    momentum = 0.5
    save_model = True

    torch.manual_seed(seed)

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

    model = Inception_BN().to(device)

    epoch_start = 1
    for epochs,lr in zip(epochs_cfg,lr_cfg):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        epoch_end = epoch_start+epochs
        for epoch in range(epoch_start,epoch_end):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
        epoch_start += epochs
    
    if save_model:
        if not osp.exists('ckpt'):
            os.makedirs('ckpt')
        torch.save(model.state_dict(), 'ckpt/cifar10_Inception_BN.pt')