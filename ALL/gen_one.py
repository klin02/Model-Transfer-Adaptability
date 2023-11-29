from model import Model
from dataloader import DataLoader
from generator import Generator,Generator_imagenet
from robust_utils import build_gen_loader, get_adversary, test_robust

import torch.nn.functional as F
import argparse
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import sys
import os
import os.path as osp

class GenOption(object):
    def __init__(self, args):
        self.model = args.model
        self.dataset = args.dataset
        self.batchSize = 128
        if self.dataset == "cifar10":
            self.nClasses = 10
        elif self.dataset == "cifar100":
            self.nClasses = 100
        else:
            assert False, "invalid dataset"

        # ----------Generator options ---------------------------------------------
        #每个epoch训练多少轮，和batchsize无关
        self.iters = 200
        #冻结embedding层权重
        self.freeze = args.freeze
        # self.randemb = args.randemb
        self.randemb = False
        # 如果不为randomemb，需要根据weight_t调整
        self.latent_dim = 64
        # 针对imagenet等数据集需要调整
        self.img_size = 32
        self.channels = 3

        
        # self.milestones_G = [40,60,80]
        if self.dataset == 'cifar10':
            self.lr_G = 0.001
            self.nEpochs = 20
            self.milestones_G = [15]
        elif self.dataset == 'cifar100':
            # self.lr_G = 0.005
            # self.nEpochs = 50
            # self.milestones_G = [20] 
            self.lr_G = 0.005
            self.nEpochs = 20
            self.milestones_G = [15]
        self.gamma_G = 0.2

        self.b1 = 0.5
        self.b2 = 0.999

        # 对抗攻击相关信息
        # 生成伪数据集的轮数，总样本量为batchSize*gen_iters
        # 减少iters以加快训练速度，此处模糊估计即可
        self.gen_iters = 20
        self.eps = 8/255
        self.steps = 10

        # ----------More option ---------------------------------------------
        self.multi_label_num = args.multi_label_num

        self.no_DM = args.no_DM
        self.noise_scale = args.noise_scale

        self.intermediate_dim = 100
        
        self.adjust = args.adjust

        self.teacher_file = 'ckpt_full/'+self.dataset+'/'+self.model+'.pt'
        gen_path = 'ckpt_gen/'+self.dataset+'/'
        self.gen_file = gen_path+ self.model+'.pt'
        self.gen_file_adjust = gen_path+self.model+'_adjust.pt'

        if not osp.exists(self.teacher_file):
            assert False, "Empty teacher file"
        if not osp.exists(gen_path):
            os.makedirs(gen_path)


class GenTrainer(object):
    def __init__(self, option):
        self.settings = option
        self.set_test_loader()
        self.set_teacher()
        self.set_generator()
        self.set_optim_G()
    
    def set_test_loader(self):
        dataloader = DataLoader(self.settings.dataset,self.settings.batchSize)
        _,_,self.test_loader = dataloader.getloader()
    
    def set_teacher(self):
        self.teacher = Model(self.settings.model,self.settings.dataset).cuda()
        self.teacher.load_state_dict(torch.load(self.settings.teacher_file))
        self.teacher.eval()
        # 用于对抗攻击使用，防止无用的hook，节省显存
        self.teacher_nohook = Model(self.settings.model,self.settings.dataset).cuda()
        self.teacher_nohook.load_state_dict(torch.load(self.settings.teacher_file))
        self.teacher_nohook.eval()

    #当randemb为False，此处同时完成了latent_dim的修改
    def set_generator(self):
        if self.settings.randemb:
            weight_t = None
        else:
            #这里要clone，防止生成器更新改变模型权重
            weight_t = self.teacher.get_output_layer_weight().clone().detach()
            # 输出层如果是Conv，weight的shape有四元，后二元都是1，舍弃
            if self.settings.model in ['Inception_BN']:
                weight_t = weight_t.reshape(weight_t.size()[:2])
            self.settings.latent_dim = weight_t.size()[1]

        if self.settings.dataset in ['cifar10','cifar100']:
            self.generator = Generator(self.settings, weight_t, self.settings.freeze).cuda()
        elif self.settings.dataset in ['imagenet']:
            self.generator = Generator_imagenet(self.settings, weight_t, self.settings.freeze).cuda()
        else:
            assert False, "Invalid dataset"

    def set_optim_G(self):
        self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
                                            betas=(self.settings.b1, self.settings.b2))
        self.lrs_G = MultiStepLR(self.optim_G, milestones=self.settings.milestones_G, gamma=self.settings.gamma_G)
    
    def test_teacher(self):
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data,target = data.cuda(), target.cuda()
                output = self.teacher(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_acc = 100. * correct / len(self.test_loader.dataset)
        print('Teacher Accuracy: {:.2f}%'.format(test_acc))
        return test_acc
        
    def prepare_train(self):
        self.log_soft = nn.LogSoftmax(dim=1)
        # MSE主要用于回归问题训练
        self.MSE_loss = nn.MSELoss().cuda()
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.teacher.eval()
        self.generator.train()
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(self.hook_fn_forward)
        #初始关闭hook，只在跟踪BN层使用
        # self.hook_switch(hook_on=False)

    # 跟踪全精度模型的BN层，提取数据分布信息
    def hook_fn_forward(self, module, input, output):
        #mean和var是针对输入的伪数据，running_mean和running_var是对train时输入的数据
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        #eval状态，直接提取QConvBN层中BN的信息即可
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)
    
    # 训练，只取类内样本
    def train(self, epoch):
        start_time = time.time()
        correct = 0
        item_len = 0
        # if epoch > 20:
        # adversary = get_adversary(self.teacher_nohook, attack_type='pgd', c=self.settings.eps, num_classes=self.settings.nClasses)
        for i in range(self.settings.iters):
            # multi_class = torch.rand(1)
            # if multi_class <0.4:
            #     MERGE_PARAM = self.settings.multi_label_num
            #     z = torch.randn(self.settings.batchSize, MERGE_PARAM,self.settings.latent_dim).cuda()
            #     labels = torch.randint(0, self.settings.nClasses, (self.settings.batchSize,MERGE_PARAM)).cuda()
            #     linear = F.softmax(torch.randn(self.settings.batchSize,MERGE_PARAM),dim=1).cuda()
            #     z = z.contiguous()
            #     labels = labels.contiguous()
            #     labels_loss = torch.zeros(self.settings.batchSize,self.settings.nClasses).cuda()
            #     labels_loss.scatter_add_(1,labels,linear)

            #     images = self.generator(z, labels, linear)
            # else:
            z = torch.randn(self.settings.batchSize, self.settings.latent_dim).cuda()
            labels = torch.randint(0, self.settings.nClasses, (self.settings.batchSize,)).cuda()
            z = z.contiguous()
            labels = labels.contiguous()
            images = self.generator(z, labels)
            # if multi_class > 0.4 and multi_class < 0.41:
            #     adv_images = images.clone()
            #     adv_images.data = adversary.perturb(images, labels)
            #     images.data = adversary.perturb(images, labels)
            labels_loss = torch.zeros(self.settings.batchSize,self.settings.nClasses).cuda()

            labels_loss.scatter_(1,labels.unsqueeze(1),1.0)
        
            self.mean_list.clear()
            self.var_list.clear()

            # 获取teacher模型输出，同时使用hook_fn_forward获取了mean和var列表
            # self.hook_switch(hook_on=True)
            output_teacher_batch = self.teacher(images)
            # self.hook_switch(hook_on=False)

            # teacher模型输出和label的损失，一维tensor
            loss_one_hot = (-(labels_loss*self.log_soft(output_teacher_batch)).sum(dim=1)).mean() 				

            # BN statistic loss
            # 这里统计了伪数据分布和teacher模型BN层分布的loss
            BNS_loss = torch.zeros(1).cuda()

            for num in range(len(self.mean_list)):
                BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
                    self.var_list[num], self.teacher_running_var[num])

            BNS_loss = BNS_loss / len(self.mean_list)

            # if epoch > 20 and not multi_class < 0:
            # if multi_class > 0.4 and multi_class < 0.45:
            #     adv_output_teacher = self.teacher_nohook(adv_images)
            #     adv_loss_one_hot = (-(labels_loss*self.log_soft(adv_output_teacher)).sum(dim=1)).mean()
            #     loss_G = loss_one_hot + 0.2 * adv_loss_one_hot + 0.1 * BNS_loss
            #     # loss_G = adv_loss_one_hot + 0.1 * BNS_loss
            #     # if i > 198:
            #     #     print("%f %f %f"%(loss_one_hot.item(),adv_loss_one_hot.item(),BNS_loss.item()))
            # else:
            #     # loss of Generator
            #     loss_G = loss_one_hot + 0.1 * BNS_loss 
            #     # if i > 198:
            #     #     print("%f %f"%(loss_one_hot.item(),BNS_loss.item()))

            loss_G = loss_one_hot + 0.1 * BNS_loss 

            self.optim_G.zero_grad()
            loss_G.backward()
            self.optim_G.step()

            # if multi_class >= 0.4:
            pred = output_teacher_batch.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            item_len += self.settings.batchSize

        gen_acc = 100. * correct / item_len
        time_interval = time.time()-start_time
        print('>> Epoch:%d  Time:%.2fs  Gen acc:%.4f'%(epoch,time_interval,gen_acc))
        return gen_acc
    
    # 根据teacher对抗下精度指导类间样本混杂
    def adjust(self):
        # 获取teacher在测试集对抗下精度
        teacher_org_acc = test_robust(self.teacher_nohook, attack_type=None, c=self.settings.eps, num_classes=self.settings.nClasses,
                            testloader=self.test_loader)
        teacher_fgsm_acc = test_robust(self.teacher_nohook, attack_type='fgsm', c=self.settings.eps, num_classes=self.settings.nClasses,
                            testloader=self.test_loader)
        teacher_pgd_acc = test_robust(self.teacher_nohook, attack_type='pgd', c=self.settings.eps, num_classes=self.settings.nClasses,
                            testloader=self.test_loader)
        gen_fgsm_acc = None
        gen_pgd_acc = None
        adjust_gen_acc = None
        iter_cnt = 0

        org_bound = 3
        bound = 1
        tolerance_cnt = 0
        tolerance_max = 300
        adversary = get_adversary(self.teacher_nohook, attack_type='pgd', c=self.settings.eps, num_classes=self.settings.nClasses)
        while True:
            # 获取teacher在伪数据集上对抗精度
            gen_loader = build_gen_loader( generator=self.generator,
                                           batchSize=self.settings.batchSize,
                                           iters=self.settings.gen_iters,
                                           latent_dim=self.settings.latent_dim,
                                           nClasses=self.settings.nClasses)
            print('>>Iters: %d'%iter_cnt)
            print('Teacher org acc: %.4f fgsm acc: %.4f pgd acc: %.4f'%(teacher_org_acc, teacher_fgsm_acc, teacher_pgd_acc))
            gen_org_acc = test_robust(self.teacher_nohook, attack_type=None, c=self.settings.eps, num_classes=self.settings.nClasses,
                                testloader=gen_loader)
            gen_fgsm_acc = test_robust(self.teacher_nohook, attack_type='fgsm', c=self.settings.eps, num_classes=self.settings.nClasses,
                                testloader=gen_loader)
            gen_pgd_acc = test_robust(self.teacher_nohook, attack_type='pgd', c=self.settings.eps, num_classes=self.settings.nClasses,
                                testloader=gen_loader)
            
            # 对抗精度限制区间
            if tolerance_cnt > tolerance_max:
                org_bound += 1
                bound += 1
                tolerance_cnt = 0
            else:
                tolerance_cnt += 1

            # 对抗精度过高，混入类间样本，过低补充类内样本，一次一个iters
            # 期望对抗精度均为testloader精度+-bound
            org_diff = gen_org_acc - teacher_org_acc
            fgsm_diff = gen_fgsm_acc - teacher_fgsm_acc
            pgd_diff = gen_pgd_acc - teacher_pgd_acc
            # if (fgsm_diff > bound and pgd_diff > -bound) or (fgsm_diff > -bound and pgd_diff > bound):
            #     multi_label = True
            # elif (fgsm_diff < bound and pgd_diff < -bound) or (fgsm_diff < -bound and pgd_diff < bound):
            #     multi_label = False
            # elif (fgsm_diff > bound and pgd_diff < -bound) or (fgsm_diff < -bound and pgd_diff > bound):
            #     if fgsm_diff + pgd_diff > 0:
            #         multi_label = True
            #     else:
            #         multi_label = False
            # if pgd_diff > bound:
            #     multi_label = True
            # elif pgd_diff < -bound:
            #     multi_label = False
            # else:
            #     # 获取最终伪数据集上的gen_acc, None表示使用纯净样本
            #     # print('>>Final Gen acc:')
            #     # adjust_gen_acc = test_robust(self.teacher, attack_type=None, c=self.settings.eps, num_classes=self.settings.nClasses,
            #     #                     testloader=gen_loader)
            #     break
            
            if org_diff < -org_bound:
                multi_label = False
                if pgd_diff < -bound:
                    adv_flag = True
                else:
                    adv_flag = False
            else:
                if pgd_diff < -bound:
                    multi_label = False
                    adv_flag = True
                elif pgd_diff > bound:
                    multi_label = True
                    adv_flag = False
                else:
                    break

            if multi_label:
                print('>>Multi Label: decrease gen_adv_acc')
                MERGE_PARAM = self.settings.multi_label_num
                z = torch.randn(self.settings.batchSize, MERGE_PARAM,self.settings.latent_dim).cuda()
                labels = torch.randint(0, self.settings.nClasses, (self.settings.batchSize,MERGE_PARAM)).cuda()
                linear = F.softmax(torch.randn(self.settings.batchSize,MERGE_PARAM),dim=1).cuda()
                z = z.contiguous()
                labels = labels.contiguous()
                labels_loss = torch.zeros(self.settings.batchSize,self.settings.nClasses).cuda()
                labels_loss.scatter_add_(1,labels,linear)
                images = self.generator(z, labels, linear)
            else:
                print('>>Single Label: rise gen_acc')
                z = torch.randn(self.settings.batchSize, self.settings.latent_dim).cuda()
                labels = torch.randint(0, self.settings.nClasses, (self.settings.batchSize,)).cuda()
                z = z.contiguous()
                labels = labels.contiguous()
                images = self.generator(z, labels)
                if adv_flag:
                    adv_images = images.clone()
                    adv_images.data = adversary.perturb(images, labels)
                labels_loss = torch.zeros(self.settings.batchSize,self.settings.nClasses).cuda()

                labels_loss.scatter_(1,labels.unsqueeze(1),1.0)
            
            self.mean_list.clear()
            self.var_list.clear()

            # 获取teacher模型输出，同时使用hook_fn_forward获取了mean和var列表
            output_teacher_batch = self.teacher(images)

            # teacher模型输出和label的损失，一维tensor
            loss_one_hot = (-(labels_loss*self.log_soft(output_teacher_batch)).sum(dim=1)).mean() 				

            # BN statistic loss
            # 这里统计了伪数据分布和teacher模型BN层分布的loss
            BNS_loss = torch.zeros(1).cuda()

            for num in range(len(self.mean_list)):
                BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
                    self.var_list[num], self.teacher_running_var[num])

            BNS_loss = BNS_loss / len(self.mean_list)

            # loss of Generator
            # if not multi_label:
            if adv_flag:
                adv_output_teacher = self.teacher_nohook(adv_images)
                adv_loss_one_hot = (-(labels_loss*self.log_soft(adv_output_teacher)).sum(dim=1)).mean()
                loss_G = loss_one_hot + 0.5 * adv_loss_one_hot + 0.1 * BNS_loss
            else:
                loss_G = loss_one_hot + 0.1 * BNS_loss 

            self.optim_G.zero_grad()
            loss_G.backward()
            self.optim_G.step()

            iter_cnt += 1
        
        # print('== Adjust: iters:%d gen_acc:%.4f FGSM_acc:%.4f/%.4f PGD_acc:%.4f/%.4f'%(iter_cnt, adjust_gen_acc, gen_fgsm_acc, teacher_fgsm_acc, gen_pgd_acc, teacher_pgd_acc))
        # return iter_cnt, adjust_gen_acc, gen_fgsm_acc, teacher_fgsm_acc, gen_pgd_acc, teacher_pgd_acc
        
        print('== Adjust: iters:%d Org_acc:%.4f/%.4f FGSM_acc:%.4f/%.4f PGD_acc:%.4f/%.4f'%(iter_cnt, gen_org_acc, teacher_org_acc, gen_fgsm_acc, teacher_fgsm_acc, gen_pgd_acc, teacher_pgd_acc))
        return iter_cnt, gen_org_acc, gen_fgsm_acc, teacher_fgsm_acc, gen_pgd_acc, teacher_pgd_acc
    
    def run(self, ft=None): # ft记录信息，可选
        test_acc = self.test_teacher()
        if ft is not None:
            ft.write('Test Result:\n')
            ft.write('\tTeacher Accuray: %f\n'%test_acc)
            ft.flush()
        
        start_time = time.time()
        self.prepare_train()
        for epoch in range(1,self.settings.nEpochs+1):
            # self.generator.train()
            gen_acc = self.train(epoch)
            # self.generator.eval()
            # gen_loader = build_gen_loader( generator=self.generator,
            #                                batchSize=self.settings.batchSize,
            #                                iters=self.settings.gen_iters,
            #                                latent_dim=self.settings.latent_dim,
            #                                nClasses=self.settings.nClasses)
            # test_robust(self.teacher_nohook, attack_type='fgsm', c=self.settings.eps, num_classes=self.settings.nClasses,
            #             testloader=gen_loader)
            # test_robust(self.teacher_nohook, attack_type='pgd', c=self.settings.eps, num_classes=self.settings.nClasses,
            #             testloader=gen_loader)
        torch.save(self.generator, self.settings.gen_file)
        train_interval = time.time()-start_time
        if ft is not None:
            ft.write('Train Result:\n')
            ft.write('\tGen acc: %f\n'%gen_acc)
            ft.write('\tTime: %.2fs\n'%train_interval)
            ft.flush()
        
        if self.settings.adjust:
            start_time = time.time()
            iter_cnt, adjust_gen_acc, gen_fgsm_acc, teacher_fgsm_acc, gen_pgd_acc, teacher_pgd_acc = self.adjust()
            torch.save(self.generator, self.settings.gen_file_adjust)
            adjust_interval = time.time()-start_time
            if ft is not None:
                ft.write('Adjust Result:\n')
                ft.write('\tIters: %d\n'%iter_cnt)
                ft.write('\tGen  acc: %f\n'%adjust_gen_acc)
                ft.write('\tFGSM acc: %f -- %f(Testloader)\n'%(gen_fgsm_acc,teacher_fgsm_acc))
                ft.write('\tPGD  acc: %f -- %f(Testloader)\n'%(gen_pgd_acc,teacher_pgd_acc))
                ft.write('\tTime: %.2fs\n'%adjust_interval)
                ft.flush()

def main():
    #及时打印信息
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='Gen Arg')
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset',type=str)

    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--randemb', action='store_true')
    parser.add_argument('--multi_label_prob', type=float, default=0.0)
    parser.add_argument('--multi_label_num', type=int, default=2)

    parser.add_argument('--no_DM', action='store_false')
    parser.add_argument('--noise_scale', type=float, default=1.0)

    # 是否使用根据全精度模型在测试集上对抗精度调整生成器
    parser.add_argument('--adjust', action='store_true')

    args = parser.parse_args()
    print(args)

    option = GenOption(args)
    print('>'*20 + 'Gen: '+option.model+'<'*20)
    
    gen_result_dir = 'gen_result/'+option.dataset+'/'
    os.makedirs(gen_result_dir, exist_ok=True)
    txt_path = gen_result_dir+option.model+'.txt'
    ft = open(txt_path,'w')

    ft.write('Gen: '+option.model+' '+option.dataset+'\n')
    ft.flush()
    print(args, file=ft)
    ft.flush()
    gentrainer = GenTrainer(option)
    gentrainer.run(ft)


if __name__ == '__main__':
    main()
