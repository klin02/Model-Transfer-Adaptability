import torch
import torch.nn as nn
import time 

from autoattack import AutoAttack

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack,\
                               LinfPGDAttack, LinfMomentumIterativeAttack, \
                               CarliniWagnerL2Attack, JacobianSaliencyMapAttack, ElasticNetL1Attack 

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index): 
        image = self.images[index]
        label = self.labels[index]
        return image, label
    

# 构建生成器的伪数据集
def build_gen_loader(generator, batchSize, iters, latent_dim, nClasses):
    gen_images = []
    gen_labels = []
    for i in range(iters):
        z = torch.randn(batchSize, latent_dim).cuda()
        labels = torch.randint(0, nClasses, (batchSize,)).cuda()
        z = z.contiguous()
        labels = labels.contiguous()
        images = generator(z, labels).detach()
        gen_images.append(images)
        gen_labels.append(labels)
    gen_images = torch.cat(gen_images)
    gen_labels = torch.cat(gen_labels)
    gen_dataset = MyDataset(gen_images, gen_labels)
    gen_loader = torch.utils.data.DataLoader(gen_dataset, batch_size=batchSize, shuffle=True)
    return gen_loader

def test_autoattack(model, testloader, norm='Linf', eps=8/255, version='standard', verbose=True):
    start_time = time.time()

    adversary = AutoAttack(model, norm=norm, eps=eps, version=version, verbose=verbose)

    if version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
        adversary.apgd.n_restarts = 1
        adversary.apgd_targeted.n_restarts = 1

    x_test = [x for (x,y) in testloader]
    x_test = torch.cat(x_test, 0)
    y_test = [y for (x,y) in testloader]
    y_test = torch.cat(y_test, 0)


    with torch.no_grad():
        x_adv, y_adv = adversary.run_standard_evaluation(x_test, y_test, bs=testloader.batch_size, return_labels=True)

    adv_correct = torch.sum(y_adv==y_test).data
    total = y_test.shape[0]

    rob_acc = adv_correct / total
    timeinterval = time.time()-start_time
    print('Attack Strength:%.4f \t  AutoAttack Acc:%.3f (%d/%d)\t Time:%.2fs'%(eps, rob_acc, adv_correct, total, timeinterval))
    
def get_adversary(model, attack_type, c, num_classes, loss_fn=nn.CrossEntropyLoss()):
    if (attack_type == "pgd"):
        adversary = LinfPGDAttack(
            model, loss_fn=loss_fn, eps=c,
            nb_iter=10, eps_iter=c/4, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False)
    elif (attack_type == "fgsm"):
        adversary = GradientSignAttack(
            model, loss_fn=loss_fn, eps=c,
            clip_min=0., clip_max=1., targeted=False)
    elif (attack_type == "mim"):
        adversary = LinfMomentumIterativeAttack(
            model, loss_fn=loss_fn, eps=c,
            nb_iter=40, eps_iter=c/10, clip_min=0., clip_max=1.,
            targeted=False)
    elif (attack_type == "bim"):
        adversary = LinfBasicIterativeAttack(
            model, loss_fn=loss_fn, eps=c,
            nb_iter=40, eps_iter=c/10, clip_min=0., clip_max=1.,
            targeted=False)
    elif (attack_type == "ela"):
        adversary = ElasticNetL1Attack(
            model, initial_const=c, confidence=0.1, max_iterations=100, clip_min=0., clip_max=1.,
            targeted=False, num_classes=10)
    elif (attack_type == "jsma"):
        adversary = JacobianSaliencyMapAttack(
            model, clip_min=0., clip_max=1., num_classes=10, gamma=c)
    elif (attack_type == "cw"):
        adversary = CarliniWagnerL2Attack(
                        model, confidence=0.01, max_iterations=1000, clip_min=0., clip_max=1., learning_rate=0.01,
                        targeted=False, num_classes=num_classes, binary_search_steps=1, initial_const=c)
    elif (attack_type == None):
        adversary = None
    else:
        raise NotImplementedError
    return adversary

def test_robust(model, attack_type, c, num_classes, testloader, loss_fn=nn.CrossEntropyLoss(), is_return=True):

    start_time = time.time()
    adversary = get_adversary(model, attack_type, c, num_classes, loss_fn)

    # ori_correct = 0
    adv_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # if batch_idx < int(req_count/testloader.batch_size):
        inputs, targets = inputs.cuda(), targets.cuda()
        total += targets.size(0)

        # ori_outputs = adversary.predict(inputs)
        # ori_preds = ori_outputs.max(dim=1, keepdim=False)[1]
        # ori_correct += ori_preds.eq(targets.data).cpu().sum()
        # nat_acc = 100. * float(ori_correct) / total
        
        if attack_type is None: #纯净样本
            advs = inputs
            with torch.no_grad():
                adv_outputs = model(advs)
        else:
            advs = adversary.perturb(inputs, targets).detach()
            with torch.no_grad():
                adv_outputs = adversary.predict(advs)
        adv_preds = adv_outputs.max(dim=1, keepdim=False)[1]
        adv_correct += adv_preds.eq(targets.data).cpu().sum()
    rob_acc = 100. * float(adv_correct) / total

    timeinterval = time.time() - start_time
    print('Attack Strength:%.4f Acc:%.3f (%d/%d) Time:%.2fs'%(c, rob_acc, adv_correct, total, timeinterval))

    if is_return:
        return rob_acc
        # return nat_acc, rob_acc
    