from dataloader import DataLoader
from robust_utils import get_adversary
from model import Model
from torch.utils.data import DataLoader 
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
import sys
import numpy as np
import torch

# CIFAR10提取的图像只有32x32像素，较为模糊，为更清晰显示，下载后进行了resize
# 使用ResNet18进行对抗攻击
if __name__ == "__main__":
    adv_dir = 'org_adv_image/'
    os.makedirs(adv_dir,exist_ok='True')
    model = 'ResNet_18'
    dataset = 'cifar10'
    batchSize = 1
    attack_type = 'pgd'
    eps = 8/255
    # dataloader = DataLoader(dataset,batchSize)
    # _,_,test_loader = dataloader.getloader()
    # 这里不需要正则化了
    test_loader = DataLoader(
        dsets.CIFAR10(root='/lustre/datasets/CIFAR10',
                      train=False,
                      transform=transforms.ToTensor(),
                      download=False),
        batch_size = batchSize,
        shuffle = True
    )

    model = Model(model,dataset).cuda()
    adversary = get_adversary(model, attack_type, eps, num_classes=10)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for inputs,targets in test_loader:
        # org
        for image, label in zip(inputs,targets):
            print(classes[label])
            torchvision.utils.save_image(image,adv_dir+'img_org.png')
        inputs,targets = inputs.cuda(), targets.cuda()
        advs = adversary.perturb(inputs, targets).detach()
        for image in advs:
            image = image.cpu()
            torchvision.utils.save_image(image,adv_dir+'img_adv.png')
        # 只取第一个图片
        sys.exit()