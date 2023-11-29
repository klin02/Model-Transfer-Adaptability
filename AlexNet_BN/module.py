import math
import numpy as np
import gol
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from function import FakeQuantize


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='sum')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output)/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2


# 获取最近的量化值
def get_nearest_val(quant_type,x,is_bias=False):
    if quant_type=='INT':
        return x.round_()
    
    plist = gol.get_value(is_bias)
    # print('get')
    # print(plist)
    shape = x.shape
    xhard = x.view(-1)
    plist = plist.type_as(x)
    # 取最近幂次作为索引
    idx = (xhard.unsqueeze(0) - plist.unsqueeze(1)).abs().min(dim=0)[1]
    xhard = plist[idx].view(shape)
    xout = (xhard - x).detach() + x
    return xout

# 采用对称有符号量化时，获取量化范围最大值
def get_qmax(quant_type,num_bits=None, e_bits=None):
    if quant_type == 'INT':
        qmax = 2. ** (num_bits - 1) - 1
    elif quant_type == 'POT':
        qmax = 1
    else: #FLOAT
        m_bits = num_bits - 1 - e_bits
        dist_m = 2 ** (-m_bits)
        e = 2 ** (e_bits - 1)
        expo = 2 ** e
        m = 2 ** m_bits -1
        frac = 1. + m * dist_m
        qmax = frac * expo
    return qmax

# 都采用有符号量化，zeropoint都置为0
def calcScaleZeroPoint(min_val, max_val, qmax):
    scale = torch.max(max_val.abs(),min_val.abs()) / qmax
    zero_point = torch.tensor(0.)
    return scale, zero_point

# 将输入进行量化，输入输出都为tensor
def quantize_tensor(quant_type, x, scale, zero_point, qmax, is_bias=False):
    # 量化后范围，直接根据位宽确定
    qmin = -qmax
    q_x = zero_point + x / scale

    q_x.clamp_(qmin, qmax)
    q_x = get_nearest_val(quant_type, q_x, is_bias)
    return q_x

# bias使用不同精度，需要根据量化类型指定num_bits/e_bits
def bias_qmax(quant_type):
    if quant_type == 'INT':
        return get_qmax(quant_type, 64)
    elif quant_type == 'POT':
        return get_qmax(quant_type)
    else:
        return get_qmax(quant_type, 16, 7)
        

# 转化为FP32，不需再做限制
def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)


class QParam(nn.Module):

    def __init__(self,quant_type, num_bits=8, e_bits=3):
        super(QParam, self).__init__()
        self.quant_type = quant_type
        self.num_bits = num_bits
        self.e_bits = e_bits
        self.qmax = get_qmax(quant_type, num_bits, e_bits)

        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        # 通过注册为register，使得buffer可以被记录到state_dict
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('min', min)
        self.register_buffer('max', max)

    # 更新统计范围及量化参数
    def update(self, tensor):
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)

        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)

        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.qmax)

    def quantize_tensor(self, tensor):
        return quantize_tensor(self.quant_type, tensor, self.scale, self.zero_point, self.qmax)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    # 该方法保证了可以从state_dict里恢复
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        key_names = ['scale', 'zero_point', 'min', 'max']
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    # 该方法返回值将是打印该对象的结果
    def __str__(self):
        info = 'scale: %.10f ' % self.scale
        info += 'zp: %.6f ' % self.zero_point
        info += 'min: %.6f ' % self.min
        info += 'max: %.6f' % self.max
        return info


# 作为具体量化层的父类，qi和qo分别为量化输入/输出
class QModule(nn.Module):

    def __init__(self,quant_type, qi=True, qo=True, num_bits=8, e_bits=3):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(quant_type,num_bits, e_bits)
        if qo:
            self.qo = QParam(quant_type,num_bits, e_bits)
        self.quant_type = quant_type
        self.num_bits = num_bits
        self.e_bits = e_bits
        self.bias_qmax = bias_qmax(quant_type)

    def freeze(self):
        pass  # 空语句

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')


"""
QModule  量化卷积

:quant_type: 量化类型
:conv_module: 卷积模块
:qi: 是否量化输入特征图
:qo: 是否量化输出特征图
:num_bits: 8位bit数
"""


class QConv2d(QModule):

    def __init__(self, quant_type, conv_module, qi=True, qo=True, num_bits=8, e_bits=3):
        super(QConv2d, self).__init__(quant_type, qi, qo, num_bits, e_bits)
        self.conv_module = conv_module
        self.qw = QParam(quant_type, num_bits,e_bits)
        self.register_buffer('M', torch.tensor([], requires_grad=False))  # 将M注册为buffer

    # freeze方法可以固定真量化的权重参数，并将该值更新到原全精度层上，便于散度计算
    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        # 这里因为在池化或者激活的输入，不需要对最大值和最小是进行额外的统计，会共享相同的输出
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        # 根据https://zhuanlan.zhihu.com/p/156835141, 这是式3 的系数
        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data

        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        self.conv_module.bias.data = quantize_tensor(self.quant_type,
                                                    self.conv_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                    zero_point=0.,qmax=self.bias_qmax, is_bias=True)
        
    def forward(self, x):  # 前向传播,输入张量,x为浮点型数据
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)  # 对输入张量X完成量化

        # foward前更新qw，保证量化weight时候scale正确
        self.qw.update(self.conv_module.weight.data)
        # 注意:此处主要为了统计各层x和weight范围，未对bias进行量化操作
        tmp_wgt = FakeQuantize.apply(self.conv_module.weight, self.qw)
        x = F.conv2d(x, tmp_wgt, self.conv_module.bias,
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation,
                     groups=self.conv_module.groups)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    # 利用公式 q_a = M(\sigma(q_w-Z_w)(q_x-Z_x) + q_b)
    def quantize_inference(self, x):  # 此处input为已经量化的qx
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x

        x = get_nearest_val(self.quant_type,x)
        x = x + self.qo.zero_point
        return x


class QLinear(QModule):

    def __init__(self, quant_type, fc_module, qi=True, qo=True, num_bits=8, e_bits=3):
        super(QLinear, self).__init__(quant_type, qi, qo, num_bits, e_bits)
        self.fc_module = fc_module
        self.qw = QParam(quant_type, num_bits, e_bits)
        self.register_buffer('M', torch.tensor([], requires_grad=False))  # 将M注册为buffer

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data

        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point
        self.fc_module.bias.data = quantize_tensor(self.quant_type,
                                                    self.fc_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                    zero_point=0., qmax=self.bias_qmax, is_bias=True)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.fc_module.weight.data)
        tmp_wgt = FakeQuantize.apply(self.fc_module.weight, self.qw)
        x = F.linear(x, tmp_wgt, self.fc_module.bias)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.fc_module(x)
        x = self.M * x
        x = get_nearest_val(self.quant_type,x)
        x = x + self.qo.zero_point

        return x


class QReLU(QModule):

    def __init__(self,quant_type, qi=False, qo=True, num_bits=8, e_bits=3):
        super(QReLU, self).__init__(quant_type, qi, qo, num_bits, e_bits)

    def freeze(self, qi=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.relu(x)

        return x

    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x


class QMaxPooling2d(QModule):

    def __init__(self, quant_type, kernel_size=3, stride=1, padding=0, qi=False, qo=True, num_bits=8,e_bits=3):
        super(QMaxPooling2d, self).__init__(quant_type, qi, qo, num_bits, e_bits)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

        return x

    def quantize_inference(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

class QConvBNReLU(QModule):

    def __init__(self, quant_type, conv_module, bn_module, qi=True, qo=True, num_bits=8, e_bits=3):
        super(QConvBNReLU, self).__init__(quant_type, qi, qo, num_bits, e_bits)
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.qw = QParam(quant_type, num_bits,e_bits)
        self.register_buffer('M', torch.tensor([], requires_grad=False))  # 将M注册为buffer

    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.view(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
            
        return weight, bias

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data

        std = torch.sqrt(self.bn_module.running_var + self.bn_module.eps)

        weight, bias = self.fold_bn(self.bn_module.running_mean, std)

        self.conv_module.weight.data = self.qw.quantize_tensor(weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        self.conv_module.bias.data = quantize_tensor(self.quant_type,
                                                     bias, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0., qmax=self.bias_qmax,is_bias=True)

    def forward(self, x):

        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        if self.training:
            y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias, 
                            stride=self.conv_module.stride,
                            padding=self.conv_module.padding,
                            dilation=self.conv_module.dilation,
                            groups=self.conv_module.groups)
            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
            y = y.contiguous().view(self.conv_module.out_channels, -1) # CNHW -> C,NHW
            # mean = y.mean(1)
            # var = y.var(1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn_module.running_mean = \
                (1 - self.bn_module.momentum) * self.bn_module.running_mean + \
                self.bn_module.momentum * mean
            self.bn_module.running_var = \
                (1 - self.bn_module.momentum) * self.bn_module.running_var + \
                self.bn_module.momentum * var
        else:
            mean = Variable(self.bn_module.running_mean)
            var = Variable(self.bn_module.running_var)

        std = torch.sqrt(var + self.bn_module.eps)

        weight, bias = self.fold_bn(mean, std)

        self.qw.update(weight.data)

        x = F.conv2d(x, FakeQuantize.apply(weight, self.qw), bias, 
                stride=self.conv_module.stride,
                padding=self.conv_module.padding, dilation=self.conv_module.dilation, 
                groups=self.conv_module.groups)

        x = F.relu(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x
    
    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x = get_nearest_val(self.quant_type,x) 
        x = x + self.qo.zero_point        
        x.clamp_(min=0)
        return x
        