import torch

def build_int_list(num_bits):
    plist = [0.]
    for i in range(0,2**(num_bits-1)): 
        # i最高到0，即pot量化最大值为1
        plist.append(i)
        plist.append(-i)
    plist = torch.Tensor(list(set(plist)))
    # plist = plist.mul(1.0 / torch.max(plist))
    return plist

def std_mid_ratio(x):
    x = x.view(-1)
    std = torch.std(x)
    max = 3*std#.item()
    min = 3*(-std)#.item()
    max = torch.max(torch.abs(max),torch.abs(min))
    mid = max/2
    cond = torch.logical_and(x>=-mid,x<=mid)
    cnt = torch.sum(cond).item()
    ratio = cnt/len(x)
    return ratio

def range_mid_ratio(x):
    x = x.view(-1)
    max = torch.max(x)
    min = torch.min(x)
    max = torch.max(torch.abs(max),torch.abs(min))
    mid = max/2
    cond = torch.logical_and(x>=-mid,x<=mid)
    cnt = torch.sum(cond).item()
    ratio = cnt/len(x)
    return ratio

def pearson_corr(tensor1, tensor2):
    """
    计算tensor1和tensor2的Pearson相关系数
    """
    if torch.equal(tensor1,tensor2):
        return 1.0
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
    
    # 计算相关系数
    corr_mat = cov_mat / torch.std(tensor1, dim=0) / torch.std(tensor2, dim=0)
    pearson = torch.mean(corr_mat)
    return pearson.item()

def cos_sim(a,b):
    a = a.view(-1)
    b = b.view(-1)
    cossim = torch.cosine_similarity(a, b, dim=0, eps=1e-6)
    # cossim = (cossim-0.97)/(1-0.97)
    return cossim.item()

def kurtosis(tensor):
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    n = tensor.numel()
    fourth_moment = ((tensor - mean)**4).sum() / n
    second_moment = std**2
    kurt = (fourth_moment / second_moment**2) - 3
    return kurt.item()