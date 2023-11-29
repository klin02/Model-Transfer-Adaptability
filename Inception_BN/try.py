import torch
import torch.nn.functional as F


def pearson_corr(tensor1, tensor2):
    """
    计算tensor1和tensor2的Pearson相关系数
    """
    # 将tensor1和tensor2展平为二维
    tensor1 = tensor1.view(-1, tensor1.size(-1))
    tensor2 = tensor2.view(-1, tensor2.size(-1))
    
    # 计算tensor1和tensor2的均值
    tensor1_mean = torch.mean(tensor1, dim=0)
    tensor2_mean = torch.mean(tensor2, dim=0)
    
    # 计算centered tensor
    tensor1_c = tensor1 - tensor1_mean
    tensor2_c = tensor2 - tensor2_mean
    
    # 计算covariance matrix
    cov_mat = torch.matmul(tensor1_c.t(), tensor2_c) / (tensor1.size(0) - 1)
    print(cov_mat)
    print('----')
    # 计算相关系数
    corr_mat = cov_mat / torch.std(tensor1, dim=0) / torch.std(tensor2, dim=0)
    print(corr_mat)
    pearson = torch.mean(corr_mat)
    return pearson

def cos_sim(a,b):
    a = a.view(-1)
    b = b.view(-1)
    cossim = torch.cosine_similarity(a, b, dim=0, eps=1e-6)
    # cossim = (cossim-0.975)/(1-0.975)
    return cossim

def mid_ratio(x):
    x = x.view(-1)
    max = torch.max(x)#.item()
    min = torch.min(x)#.item()
    max = torch.max(torch.abs(max),torch.abs(min))
    mid = max/2
    cond = torch.logical_and(x>=-mid,x<=mid)
    print(cond)
    cnt = torch.sum(cond).item()
    print(cnt)
    ratio = cnt/len(x)
    print(ratio)

if __name__ == "__main__":
    # 创建两个3维的tensor
    x = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]]) 
    y = torch.tensor([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
    z = torch.tensor([-3,-2,-1,1,2,3])
    x = x.float()
    y = y.float()
    # x = F.normalize(x,p=2,dim=-1)
    # y = F.normalize(y,p=2,dim=-1)
    # 计算相关系数
    # r = pearson_corr(x, y)
    # r = cos_sim(x,y)
    mid_ratio(y)
    # 输出相关系数
    # print(r)