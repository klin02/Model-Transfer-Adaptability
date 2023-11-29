import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *
import module

# conv: 'C',''/'B'/'BR',qi,in_ch,out_ch,kernel_size,stirde,padding,
# relu: 'R'
# inception: 'Inc'
# maxpool: 'MP',kernel_size,stride,padding
# adaptiveavgpool: 'AAP',output_size
# flatten: 'F'

# class 10
Inception_BN_cfg_table = [
    ['C','',True,3,64,3,1,1],
    ['R'],
    ['C','',False,64,64,3,1,1],
    ['R'],
    ['Inc',0],
    ['Inc',1],
    ['MP',3,2,1],
    ['Inc',2],
    ['Inc',3],
    ['Inc',4],
    ['Inc',5],
    ['Inc',6],
    ['MP',3,2,1],
    ['Inc',7],
    ['Inc',8],
    ['AAP',1],
    ['C','',False,1024,10,1,1,0],
    ['F']
]
inc_ch_table=[
    [64,  64, 96,128, 16, 32, 32],#3a
    [256,128,128,192, 32, 96, 64],#3b
    [480,192, 96,208, 16, 48, 64],#4a
    [512,160,112,224, 24, 64, 64],#4b
    [512,128,128,256, 24, 64, 64],#4c
    [512,112,144,288, 32, 64, 64],#4d
    [528,256,160,320, 32,128,128],#4e
    [832,256,160,320, 32,128,128],#5a
    [832,384,192,384, 48,128,128] #5b
]

# br0,br1,br2,br3 <- br1x1,br3x3,br5x5,brM
# 这里的第2,3个参数是channel中的索引
# 对于后续cfg拓展，可以认为'C'有'BR'参数。这里的一个项根据量化后可融合结构指定
inc_cfg_table = [
    [['C',0,1,1,1,0]],
    [['C',0,2,1,1,0],
     ['C',2,3,3,1,1]],
    [['C',0,4,1,1,0],
     ['C',4,5,5,1,2]],
    [['MP',3,1,1],
     ['R'],
     ['C',0,6,1,1,0]]
]
def make_layers(model,cfg_table):
    for i in range(len(cfg_table)):
        cfg = cfg_table[i]
        if cfg[0] == 'Inc':
            make_inc_layers(model,cfg[1])
        elif cfg[0] == 'C':
            name = 'conv%d'%i
            layer = nn.Conv2d(cfg[3],cfg[4],kernel_size=cfg[5],stride=cfg[6],padding=cfg[7])
            model.add_module(name,layer)
            if 'B' in cfg[1]:
                name = 'bn%d'%i
                layer = nn.BatchNorm2d(cfg[4])
                model.add_module(name,layer)
                if 'R' in cfg[1]:
                    name = 'relu%d'%i
                    layer = nn.ReLU(True)
                    model.add_module(name,layer)
        elif cfg[0] == 'R':
            name = 'relu%d'%i
            layer = nn.ReLU(True)
            model.add_module(name,layer)
        elif cfg[0] == 'MP':
            name = 'pool%d'%i
            layer = nn.MaxPool2d(kernel_size=cfg[1],stride=cfg[2],padding=cfg[3])
            model.add_module(name,layer)
        elif cfg[0] == 'AAP':
            name = 'aap%d'%i
            layer = nn.AdaptiveAvgPool2d(cfg[1])
            model.add_module(name,layer)

def model_forward(model,cfg_table,x):
    for i in range(len(cfg_table)):
        cfg = cfg_table[i]
        if cfg[0] == 'Inc':
            x = inc_forward(model,cfg[1],x)
        elif cfg[0] == 'C':
            name = 'conv%d'%i
            layer = getattr(model,name)
            x = layer(x)
            if 'B' in cfg[1]:
                name = 'bn%d'%i
                layer = getattr(model,name)
                x = layer(x)
                if 'R' in cfg[1]:
                    name = 'relu%d'%i
                    layer = getattr(model,name)
                    x = layer(x)
        elif cfg[0] == 'R':
            name = 'relu%d'%i
            layer = getattr(model,name)
            x = layer(x)
        elif cfg[0] == 'MP':
            name = 'pool%d'%i
            layer = getattr(model,name)
            x = layer(x)
        elif cfg[0] == 'AAP':
            name = 'aap%d'%i
            layer = getattr(model,name)
            x = layer(x)
        elif cfg[0] == 'F':
            x = torch.flatten(x, start_dim=1)

    return x

def model_quantize(model,cfg_table,quant_type,num_bits,e_bits):
    for i in range(len(cfg_table)):
        cfg = cfg_table[i]
        if cfg[0] == 'Inc':
            inc_quantize(model,cfg[1],quant_type,num_bits,e_bits)
        elif cfg[0] == 'C':
            conv_name = 'conv%d'%i
            conv_layer = getattr(model,conv_name)
            qname = 'q_'+conv_name
            if 'B' in cfg[1]:
                bn_name = 'bn%d'%i
                bn_layer = getattr(model,bn_name)
                if 'R' in cfg[1]:
                    qlayer = QConvBNReLU(quant_type,conv_layer,bn_layer,qi=cfg[2],num_bits=num_bits,e_bits=e_bits)
                else:
                    qlayer = QConvBN(quant_type,conv_layer,bn_layer,qi=cfg[2],num_bits=num_bits,e_bits=e_bits)
            else:
                qlayer = QConv2d(quant_type,conv_layer,qi=cfg[2],num_bits=num_bits,e_bits=e_bits)
            model.add_module(qname,qlayer)
        elif cfg[0] == 'R':
            name = 'relu%d'%i
            qname = 'q_'+name
            qlayer = QReLU(quant_type,num_bits=num_bits,e_bits=e_bits)
            model.add_module(qname,qlayer)
        elif cfg[0] == 'MP':
            name = 'pool%d'%i
            qname = 'q_'+name
            qlayer = QMaxPooling2d(quant_type,kernel_size=cfg[1],stride=cfg[2],padding=cfg[3],num_bits=num_bits,e_bits=e_bits)
            model.add_module(qname,qlayer)
        elif cfg[0] == 'AAP':
            name = 'aap%d'%i
            qname = 'q_'+name
            qlayer = QAdaptiveAvgPool2d(quant_type,output_size=cfg[1],num_bits=num_bits,e_bits=e_bits)
            model.add_module(qname,qlayer)

# 支持选择反量化位置，进行debug。最后release时可取消
# end_pos为-1时表示到最后才反量化，否则在i层反量化 
# 增加了func='fakefreeze'
def model_utils(model,cfg_table,func,x=None):
    end_flag = False
    end_pos = -1
    last_qo = None
    for i in range(len(cfg_table)):
        cfg = cfg_table[i]
        if i == end_pos:
            end_flag = True
            if func == 'inference':
                x = last_qo.dequantize_tensor(x)
        if cfg[0] == 'Inc':
            if end_flag:
                if func == 'inference' or func == 'forward':
                    x = inc_forward(model,cfg[1],x)
                continue
            x,last_qo = inc_utils(model,cfg[1],func,x,last_qo)
        elif cfg[0] == 'C':
            if end_flag:
                name = 'conv%d'%i
                layer = getattr(model,name)
                if func == 'inference' or func == 'forward':
                    x = layer(x)
                continue
            qname = 'q_conv%d'%i
            qlayer = getattr(model,qname)
            if func == 'forward':
                x = qlayer(x)
            elif func == 'inference':
                # cfg[2]为True表示起始层，需要量化
                if cfg[2]:
                    x = qlayer.qi.quantize_tensor(x)
                x = qlayer.quantize_inference(x)
            elif func == 'freeze':
                qlayer.freeze(last_qo)
            elif func == 'fakefreeze':
                qlayer.fakefreeze()
            last_qo = qlayer.qo
        elif cfg[0] == 'R':
            if end_flag:
                name = 'relu%d'%i
                layer = getattr(model,name)
                if func == 'inference' or func == 'forward':
                    x = layer(x)
                continue
            qname = 'q_relu%d'%i
            qlayer = getattr(model,qname)
            if func == 'forward':
                x = qlayer(x)
            elif func == 'inference':
                x = qlayer.quantize_inference(x)
            elif func == 'freeze':
                qlayer.freeze(last_qo)
            elif func == 'fakefreeze':
                qlayer.fakefreeze()
        elif cfg[0] == 'MP':
            if end_flag:
                name = 'pool%d'%i
                layer = getattr(model,name)
                if func == 'inference' or func == 'forward':
                    x = layer(x)
                continue
            qname = 'q_pool%d'%i
            qlayer = getattr(model,qname)
            if func == 'forward':
                x = qlayer(x)
            elif func == 'inference':
                x = qlayer.quantize_inference(x)
            elif func == 'freeze':
                qlayer.freeze(last_qo)
            elif func == 'fakefreeze':
                qlayer.fakefreeze()
        elif cfg[0] == 'AAP':
            if end_flag:
                name = 'aap%d'%i
                layer = getattr(model,name)
                if func == 'inference' or func == 'forward':
                    x = layer(x)
                continue
            qname = 'q_aap%d'%i
            qlayer = getattr(model,qname)
            if func == 'forward':
                x = qlayer(x)
            elif func == 'inference':
                x = qlayer.quantize_inference(x)
            elif func == 'freeze':
                qlayer.freeze(last_qo)
            elif func == 'fakefreeze':
                qlayer.fakefreeze()
            last_qo = qlayer.qo
        elif cfg[0] == 'F':
            if func == 'inference' or func == 'forward':
                x = torch.flatten(x,start_dim=1)
    
    if func == 'inference' and not end_flag:
        x = last_qo.dequantize_tensor(x)

    return x

def make_inc_layers(model,inc_idx):
    inc_name = 'inc%d'%inc_idx
    ch = inc_ch_table[inc_idx]
    for i in range(4): # branch
        prefix = inc_name+'_br%d_'%i
        for j in range(len(inc_cfg_table[i])):
            cfg = inc_cfg_table[i][j]
            if cfg[0] == 'MP':
                name = prefix+'pool%d'%j
                layer =nn.MaxPool2d(kernel_size=cfg[1],stride=cfg[2],padding=cfg[3])
                model.add_module(name,layer)
            elif cfg[0] == 'R':
                name=prefix+'relu%d'%j
                layer=nn.ReLU(True)
                model.add_module(name,layer)
            elif cfg[0] == 'C':
                name=prefix+'conv%d'%j
                layer=nn.Conv2d(ch[cfg[1]],ch[cfg[2]],kernel_size=cfg[3],stride=cfg[4],padding=cfg[5],bias=False)
                model.add_module(name,layer)
                name=prefix+'bn%d'%j
                layer=nn.BatchNorm2d(ch[cfg[2]])
                model.add_module(name,layer)
                name=prefix+'relu%d'%j
                layer=nn.ReLU(True)
                model.add_module(name,layer)

def inc_forward(model,inc_idx,x=None):
    inc_name = 'inc%d'%inc_idx
    outs = []
    for i in range(4):
        prefix = inc_name+'_br%d_'%i
        tmp = x
        for j in range(len(inc_cfg_table[i])):
            cfg = inc_cfg_table[i][j]
            if cfg[0] == 'MP':
                name = prefix+'pool%d'%j
                layer = getattr(model,name)
                tmp = layer(tmp)
            elif cfg[0] == 'R':
                name=prefix+'relu%d'%j
                layer = getattr(model,name)
                tmp = layer(tmp)
            else:
                name=prefix+'conv%d'%j
                layer = getattr(model,name)
                tmp = layer(tmp)
                name=prefix+'bn%d'%j
                layer = getattr(model,name)
                tmp = layer(tmp)
                name=prefix+'relu%d'%j
                layer = getattr(model,name)
                tmp = layer(tmp)
        outs.append(tmp)
    
    out = torch.cat(outs,1)
    return out

def inc_quantize(model,inc_idx,quant_type,num_bits,e_bits):
    inc_name = 'inc%d'%inc_idx
    for i in range(4):
        prefix = inc_name+'_br%d_'%i
        for j in range(len(inc_cfg_table[i])):
            cfg = inc_cfg_table[i][j]
            if cfg[0] == 'MP':
                name = prefix+'pool%d'%j
                qname = 'q_'+name
                qlayer = QMaxPooling2d(quant_type,kernel_size=cfg[1],stride=cfg[2],padding=cfg[3],num_bits=num_bits,e_bits=e_bits)
                model.add_module(qname,qlayer)
            elif cfg[0] == 'R':
                name = prefix+'relu%d'%j
                qname = 'q_'+name
                qlayer = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
                model.add_module(qname,qlayer)
            else:
                conv_name=prefix+'conv%d'%j
                conv_layer=getattr(model,conv_name)
                bn_name=prefix+'bn%d'%j
                bn_layer=getattr(model,bn_name)
                qname='q_'+conv_name
                qlayer=QConvBNReLU(quant_type, conv_layer, bn_layer, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
                model.add_module(qname,qlayer)
    qname = 'q_'+inc_name+'_concat'
    qlayer = QConcat(quant_type,4,qi_array=False,qo=True,num_bits=num_bits,e_bits=e_bits)
    model.add_module(qname,qlayer)
    
def inc_utils(model,inc_idx,func,x=None,qo=None):
    inc_name = 'inc%d'%inc_idx
    outs=[]
    qos=[]
    for i in range(4): 
        qprefix = 'q_'+inc_name+'_br%d_'%i
        tmp = x
        last_qo = qo
        for j in range(len(inc_cfg_table[i])):
            cfg = inc_cfg_table[i][j]
            if cfg[0] == 'MP':
                qname = qprefix+'pool%d'%j
                qlayer = getattr(model,qname)
                if func == 'forward':
                    tmp = qlayer(tmp)
                elif func == 'inference':
                    tmp = qlayer.quantize_inference(tmp)
                elif func == 'freeze':
                    qlayer.freeze(last_qo)
                elif func == 'fakefreeze':
                    qlayer.fakefreeze()
            elif cfg[0] == 'R':
                qname = qprefix+'relu%d'%j
                qlayer = getattr(model,qname)
                if func == 'forward':
                    tmp = qlayer(tmp)
                elif func == 'inference':
                    tmp = qlayer.quantize_inference(tmp)
                elif func == 'freeze':
                    qlayer.freeze(last_qo)
                elif func == 'fakefreeze':
                    qlayer.fakefreeze()
            else:
                qname = qprefix+'conv%d'%j
                qlayer = getattr(model,qname)
                if func == 'forward':
                    tmp = qlayer(tmp)
                elif func == 'inference':
                    tmp = qlayer.quantize_inference(tmp)
                elif func == 'freeze':
                    qlayer.freeze(last_qo)
                elif func == 'fakefreeze':
                    qlayer.fakefreeze()
                last_qo = qlayer.qo
        
        outs.append(tmp)
        qos.append(last_qo)

    qname = 'q_'+inc_name+'_concat'
    qlayer = getattr(model,qname)
    out = None
    if func == 'forward':
        out = qlayer(outs)
    elif func == 'inference':
        out = qlayer.quantize_inference(outs)
    else: #freeze
        qlayer.freeze(qos)
    last_qo = qlayer.qo
    return out,last_qo


class Inception_BN(nn.Module):
    def __init__(self,num_classes=10):
        super(Inception_BN, self).__init__()
        self.cfg_table = Inception_BN_cfg_table
        make_layers(self,self.cfg_table)
    
    def forward(self,x):
        x = model_forward(self,self.cfg_table,x)
        return x

    def quantize(self, quant_type, num_bits=8, e_bits=3):
        model_quantize(self,self.cfg_table,quant_type,num_bits,e_bits)
    
    def quantize_forward(self,x):
        return model_utils(self,self.cfg_table,func='forward',x=x)

    def freeze(self):
        model_utils(self,self.cfg_table,func='freeze')
    
    def quantize_inference(self,x):
        return model_utils(self,self.cfg_table,func='inference',x=x)

    def fakefreeze(self):
        model_utils(self,self.cfg_table,func='fakefreeze')

if __name__ == "__main__":
    model = Inception_BN()
    model.quantize('INT',8,3)
    print(model.named_modules)
    print('-------')
    print(model.named_parameters)
    print(len(model.conv0.named_parameters()))