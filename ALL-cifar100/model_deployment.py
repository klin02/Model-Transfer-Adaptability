import torch.nn as nn
import torch.nn.functional as F
from cfg import *
from module import *

def make_layers(model,cfg_table):
    for i in range(len(cfg_table)):
        cfg = cfg_table[i]
        if cfg[0] == 'Inc':
            make_inc_layers(model,cfg[1])
        elif cfg[0] == 'ML':
            make_ml_layers(model,cfg[1],cfg[2],cfg[3])
        elif cfg[0] == 'C':
            name = 'conv%d'%i
            layer = nn.Conv2d(cfg[3],cfg[4],kernel_size=cfg[5],stride=cfg[6],padding=cfg[7],bias=cfg[8])
            model.add_module(name,layer)
            if 'B' in cfg[1]:
                name = 'bn%d'%i
                layer = nn.BatchNorm2d(cfg[4])
                model.add_module(name,layer)
                if 'RL' in cfg[1]:
                    name = 'relu%d'%i
                    layer = nn.ReLU(True)
                    model.add_module(name,layer)
                elif 'RS' in cfg[1]:
                    name = 'relus%d'%i
                    layer = nn.ReLU6(True)
                    model.add_module(name,layer)
        elif cfg[0] == 'RL':
            name = 'relu%d'%i
            layer = nn.ReLU(True)
            model.add_module(name,layer)
        elif cfg[0] == 'RS':
            name = 'relus%d'%i
            layer = nn.ReLU6(True)
            model.add_module(name,layer)
        elif cfg[0] == 'MP':
            name = 'pool%d'%i
            layer = nn.MaxPool2d(kernel_size=cfg[1],stride=cfg[2],padding=cfg[3])
            model.add_module(name,layer)
        elif cfg[0] == 'AAP':
            name = 'aap%d'%i
            layer = nn.AdaptiveAvgPool2d(cfg[1])
            model.add_module(name,layer)
        elif cfg[0] == 'FC':
            name = 'fc%d'%i
            layer = nn.Linear(cfg[1],cfg[2],bias=cfg[3])
            model.add_module(name,layer)
        elif cfg[0] == 'D':
            name = 'drop%d'%i
            layer = nn.Dropout(cfg[1])
            model.add_module(name,layer)
        

def model_forward(model,cfg_table,x):
    for i in range(len(cfg_table)):
        cfg = cfg_table[i]
        if cfg[0] == 'Inc':
            x = inc_forward(model,cfg[1],x)
        elif cfg[0] == 'ML':
            x = ml_forward(model,cfg[1],cfg[2],cfg[3],x)
        elif cfg[0] == 'C':
            name = 'conv%d'%i
            layer = getattr(model,name)
            x = layer(x)
            if 'B' in cfg[1]:
                name = 'bn%d'%i
                layer = getattr(model,name)
                x = layer(x)
                if 'RL' in cfg[1]:
                    name = 'relu%d'%i
                    layer = getattr(model,name)
                    x = layer(x)
                elif 'RS' in cfg[1]:
                    name = 'relus%d'%i
                    layer = getattr(model,name)
                    x = layer(x)
        elif cfg[0] == 'RL':
            name = 'relu%d'%i
            layer = getattr(model,name)
            x = layer(x)
        elif cfg[0] == 'RS':
            name = 'relus%d'%i
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
        elif cfg[0] == 'FC':
            name = 'fc%d'%i
            layer = getattr(model,name)
            x = layer(x)
        elif cfg[0] == 'D':
            name = 'drop%d'%i
            layer = getattr(model,name)
            x = layer(x)
        elif cfg[0] == 'VW':
            if len(cfg) == 1: #default
                x = x.view(x.size(0),-1)
        elif cfg[0] == 'SM':
            x = F.softmax(x,dim=1)
    return x


def model_quantize(model,cfg_table,quant_type,num_bits,e_bits):
    for i in range(len(cfg_table)):
        cfg = cfg_table[i]
        if cfg[0] == 'Inc':
            inc_quantize(model,cfg[1],quant_type,num_bits,e_bits)
        elif cfg[0] == 'ML':
            ml_quantize(model,cfg[1],cfg[2],cfg[3],quant_type,num_bits,e_bits)
        elif cfg[0] == 'C':
            conv_name = 'conv%d'%i
            conv_layer = getattr(model,conv_name)
            qname = 'q_'+conv_name
            if 'B' in cfg[1]:
                bn_name = 'bn%d'%i
                bn_layer = getattr(model,bn_name)
                if 'RL' in cfg[1]:
                    qlayer = QConvBNReLU(quant_type,conv_layer,bn_layer,qi=cfg[2],num_bits=num_bits,e_bits=e_bits)
                elif 'RS' in cfg[1]:
                    qlayer = QConvBNReLU6(quant_type,conv_layer,bn_layer,qi=cfg[2],num_bits=num_bits,e_bits=e_bits)
                else:
                    qlayer = QConvBN(quant_type,conv_layer,bn_layer,qi=cfg[2],num_bits=num_bits,e_bits=e_bits)
            else:
                qlayer = QConv2d(quant_type,conv_layer,qi=cfg[2],num_bits=num_bits,e_bits=e_bits)
            model.add_module(qname,qlayer)
        elif cfg[0] == 'RL':
            name = 'relu%d'%i
            qname = 'q_'+name
            qlayer = QReLU(quant_type,num_bits=num_bits,e_bits=e_bits)
            model.add_module(qname,qlayer)
        elif cfg[0] == 'RS':
            name = 'relus%d'%i
            qname = 'q_'+name
            qlayer = QReLU6(quant_type,num_bits=num_bits,e_bits=e_bits)
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
        elif cfg[0] == 'FC':
            name = 'fc%d'%i 
            layer = getattr(model,name)
            qname = 'q_'+name
            qlayer = QLinear(quant_type,layer,num_bits=num_bits,e_bits=e_bits)
            model.add_module(qname,qlayer)


# 增加了func='fakefreeze'
def model_utils(model,cfg_table,func,x=None):
    last_qo = None
    # 表示已经经过反量化，用于区别反量化不再最后，而是在softmax前的情形
    done_flag = False
    for i in range(len(cfg_table)):
        cfg = cfg_table[i]
        if cfg[0] == 'Inc':
            x,last_qo = inc_utils(model,cfg[1],func,x,last_qo)
        elif cfg[0] == 'ML':
            x,last_qo = ml_utils(model,cfg[1],cfg[2],cfg[3],func,x,last_qo)
        elif cfg[0] == 'C':
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
        elif cfg[0] == 'RL':
            qname = 'q_relu%d'%i
            qlayer = getattr(model,qname)
            if func == 'forward':
                x = qlayer(x)
            elif func == 'inference':
                x = qlayer.quantize_inference(x)
            elif func == 'freeze':
                qlayer.freeze(last_qo)
        elif cfg[0] == 'RS':
            qname = 'q_relus%d'%i
            qlayer = getattr(model,qname)
            if func == 'forward':
                x = qlayer(x)
            elif func == 'inference':
                x = qlayer.quantize_inference(x)
            elif func == 'freeze':
                qlayer.freeze(last_qo)
        elif cfg[0] == 'MP':
            qname = 'q_pool%d'%i
            qlayer = getattr(model,qname)
            if func == 'forward':
                x = qlayer(x)
            elif func == 'inference':
                x = qlayer.quantize_inference(x)
            elif func == 'freeze':
                qlayer.freeze(last_qo)
        elif cfg[0] == 'AAP':
            qname = 'q_aap%d'%i
            qlayer = getattr(model,qname)
            if func == 'forward':
                x = qlayer(x)
            elif func == 'inference':
                x = qlayer.quantize_inference(x)
            elif func == 'freeze':
                qlayer.freeze(last_qo)
            last_qo = qlayer.qo
        elif cfg[0] == 'FC':
            qname = 'q_fc%d'%i
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
        elif cfg[0] == 'D':
            if func == 'forward':
                name = 'drop%d'%i
                layer = getattr(model,name)
                x = layer(x)
        elif cfg[0] == 'VW':
            if func == 'inference' or func == 'forward':
                if len(cfg) == 1: #default
                    x = x.view(x.size(0),-1)
        elif cfg[0] == 'SM':
            if func == 'inference':
                done_flag = True
                x = last_qo.dequantize_tensor(x)
                x = F.softmax(x,dim=1)
            elif func == 'forward':
                x = F.softmax(x,dim=1)

    if func == 'inference' and not done_flag:
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
            elif cfg[0] == 'RL':
                name=prefix+'relu%d'%j
                layer=nn.ReLU(True)
                model.add_module(name,layer)
            elif cfg[0] == 'C': # 'BRL' default
                name=prefix+'conv%d'%j
                layer=nn.Conv2d(ch[cfg[1]],ch[cfg[2]],kernel_size=cfg[3],stride=cfg[4],padding=cfg[5],bias=False)
                model.add_module(name,layer)
                name=prefix+'bn%d'%j
                layer=nn.BatchNorm2d(ch[cfg[2]])
                model.add_module(name,layer)
                name=prefix+'relu%d'%j
                layer=nn.ReLU(True)
                model.add_module(name,layer)


def inc_forward(model,inc_idx,x):
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
            elif cfg[0] == 'RL':
                name=prefix+'relu%d'%j
                layer = getattr(model,name)
                tmp = layer(tmp)
            elif cfg[0] == 'C': # 'BRL' default
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
            elif cfg[0] == 'RL':
                name = prefix+'relu%d'%j
                qname = 'q_'+name
                qlayer = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
                model.add_module(qname,qlayer)
            elif cfg[0] == 'C': # 'BRL' default
                conv_name=prefix+'conv%d'%j
                conv_layer=getattr(model,conv_name)
                bn_name=prefix+'bn%d'%j
                bn_layer=getattr(model,bn_name)
                qname='q_'+conv_name
                qlayer=QConvBNReLU(quant_type, conv_layer, bn_layer, num_bits=num_bits, e_bits=e_bits)
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
            elif cfg[0] == 'RL':
                qname = qprefix+'relu%d'%j
                qlayer = getattr(model,qname)
                if func == 'forward':
                    tmp = qlayer(tmp)
                elif func == 'inference':
                    tmp = qlayer.quantize_inference(tmp)
                elif func == 'freeze':
                    qlayer.freeze(last_qo)
            elif cfg[0] == 'C': # 'BRL' default
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
    elif func == 'freeze':
        qlayer.freeze(qos)
    last_qo = qlayer.qo
    return out,last_qo



def make_ml_layers(model,blk_type,ml_idx,blocks):
    ml_name = 'ml%d'%ml_idx
    if blk_type == 'BBLK':
        blk_ch_table = bblk_ch_table
        blk_cfg_table = bblk_cfg_table
    elif blk_type == 'BTNK':
        blk_ch_table = btnk_ch_table
        blk_cfg_table = btnk_cfg_table
    elif blk_type == 'IRES':
        blk_ch_table = ires_ch_table
        blk_cfg_table = ires_cfg_table
    else:
        raise ValueError("Make_ml_layers: Illegal blk_type")
    
    #一个makelayer对应两行，分别表示第一个blk和其余的特征
    make_blk_layers(model,blk_ch_table,blk_cfg_table,ml_name,2*ml_idx,0)
    for i in range(1,blocks):
        make_blk_layers(model,blk_ch_table,blk_cfg_table,ml_name,2*ml_idx+1,i)


# ma表示主分支，ds表示downsample
# 当cfgtable含有三个元素时，第3个表示分支合并后需经过的层。
# BasicBlock和BottleNeck合并后需经过relu层，InvertedResidual合并后无需经过relu层
def make_blk_layers(model,blk_ch_table,blk_cfg_table,ml_name,ch_idx,blk_idx):
    blk_name = ml_name+'_blk%d'%blk_idx
    ch = blk_ch_table[ch_idx]
    for i in range(2):
        if i == 0:
            prefix = blk_name+'_ma_'
        elif i == 1: 
            if ch[0]: #downsample/identity_flag
                prefix = blk_name+'_ds_'
            else:
                continue
        
        for j in range(len(blk_cfg_table[i])):
            cfg = blk_cfg_table[i][j]
            if cfg[0] == 'C':
                name = prefix+'conv%d'%j
                layer = nn.Conv2d(ch[cfg[2]],ch[cfg[3]],kernel_size=cfg[4],stride=ch[cfg[5]],padding=cfg[6],groups=ch[cfg[7]])
                model.add_module(name,layer)
                if 'B' in cfg[1]:
                    name = prefix+'bn%d'%j
                    layer=nn.BatchNorm2d(ch[cfg[3]])
                    model.add_module(name,layer)
                    if 'RL' in cfg[1]:
                        name = prefix+'relu%d'%j
                        layer = nn.ReLU(True)
                        model.add_module(name,layer)
                    elif 'RS' in cfg[1]:
                        name = prefix+'relus%d'%j
                        layer = nn.ReLU6(True)
                        model.add_module(name,layer)

    #分支汇总
    prefix = blk_name+'_'
    for j in range(len(blk_cfg_table[-1])):
        cfg = blk_cfg_table[-1][j]
        if cfg[0] == 'RL': #当前没有blk出现汇总处有RS
            name = prefix+'relu%d'%j
            layer = nn.ReLU(True)
            model.add_module(name,layer)


def ml_forward(model,blk_type,ml_idx,blocks,x):
    ml_name = 'ml%d'%ml_idx
    if blk_type == 'BBLK':
        blk_ch_table = bblk_ch_table
        blk_cfg_table = bblk_cfg_table
    elif blk_type == 'BTNK':
        blk_ch_table = btnk_ch_table
        blk_cfg_table = btnk_cfg_table
    elif blk_type == 'IRES':
        blk_ch_table = ires_ch_table
        blk_cfg_table = ires_cfg_table
    else:
        raise ValueError("ml_forward: Illegal blk_type")
    
    x = blk_forward(model,blk_ch_table,blk_cfg_table,ml_name,2*ml_idx,0,x)
    for i in range(1,blocks):
        x = blk_forward(model,blk_ch_table,blk_cfg_table,ml_name,2*ml_idx+1,i,x)

    return x


def blk_forward(model,blk_ch_table,blk_cfg_table,ml_name,ch_idx,blk_idx,x):
    blk_name = ml_name+'_blk%d'%blk_idx
    ch = blk_ch_table[ch_idx]
    outs = []
    for i in range(2):
        tmp=x
        if i == 0:
            prefix = blk_name+'_ma_'
        elif i == 1: 
            if ch[0]: #downsample/identity_flag
                prefix = blk_name+'_ds_'
            else:
                outs.append(tmp)
                continue
       
        for j in range(len(blk_cfg_table[i])):
            cfg = blk_cfg_table[i][j]
            if cfg[0] == 'C':
                name = prefix+'conv%d'%j
                layer = getattr(model,name)
                tmp = layer(tmp)
                if 'B' in cfg[1]:
                    name = prefix+'bn%d'%j
                    layer = getattr(model,name)
                    tmp = layer(tmp)
                    if 'RL' in cfg[1]:
                        name = prefix+'relu%d'%j
                        layer = getattr(model,name)
                        tmp = layer(tmp)
                    elif 'RS' in cfg[1]:
                        name = prefix+'relus%d'%j
                        layer = getattr(model,name)
                        tmp = layer(tmp)
        
        outs.append(tmp)
    
    #分支汇总
    prefix = blk_name+'_'
    for j in range(len(blk_cfg_table[-1])):
        cfg = blk_cfg_table[-1][j]
        if cfg[0] == 'AD':
            if cfg[1] or ch[0]: #无条件加或flag为true
                out = outs[0] + outs[1]
            else:
                out = outs[0]
        elif cfg[0] == 'RL':
            name = prefix+'relu%d'%j
            layer = getattr(model,name)
            out = layer(out)

    return out


def ml_quantize(model,blk_type,ml_idx,blocks,quant_type,num_bits,e_bits):
    ml_name = 'ml%d'%ml_idx
    if blk_type == 'BBLK':
        blk_ch_table = bblk_ch_table
        blk_cfg_table = bblk_cfg_table
    elif blk_type == 'BTNK':
        blk_ch_table = btnk_ch_table
        blk_cfg_table = btnk_cfg_table
    elif blk_type == 'IRES':
        blk_ch_table = ires_ch_table
        blk_cfg_table = ires_cfg_table
    else:
        raise ValueError("ml_quantize: Illegal blk_type")
    
    blk_quantize(model,blk_ch_table,blk_cfg_table,ml_name,2*ml_idx,0,quant_type,num_bits,e_bits)
    for i in range(1,blocks):
        blk_quantize(model,blk_ch_table,blk_cfg_table,ml_name,2*ml_idx+1,i,quant_type,num_bits,e_bits)


def blk_quantize(model,blk_ch_table,blk_cfg_table,ml_name,ch_idx,blk_idx,quant_type,num_bits,e_bits):
    blk_name = ml_name+'_blk%d'%blk_idx
    ch = blk_ch_table[ch_idx]
    for i in range(2):
        if i == 0:
            prefix = blk_name+'_ma_'
        elif i == 1: 
            if ch[0]: #downsample/identity_flag
                prefix = blk_name+'_ds_'
            else:
                continue

        for j in range(len(blk_cfg_table[i])):
            cfg = blk_cfg_table[i][j]
            if cfg[0] == 'C':
                conv_name = prefix+'conv%d'%j
                conv_layer = getattr(model,conv_name)
                qname = 'q_'+conv_name
                if 'B' in cfg[1]:
                    bn_name = prefix+'bn%d'%j
                    bn_layer = getattr(model,bn_name)
                    if 'RL' in cfg[1]:
                        qlayer = QConvBNReLU(quant_type,conv_layer,bn_layer,num_bits=num_bits,e_bits=e_bits)
                    elif 'RS' in cfg[1]:
                        qlayer = QConvBNReLU6(quant_type,conv_layer,bn_layer,num_bits=num_bits,e_bits=e_bits)
                    else:
                        qlayer = QConvBN(quant_type,conv_layer,bn_layer,num_bits=num_bits,e_bits=e_bits)
                else:
                    qlayer = QConv2d(quant_type,conv_layer,num_bits=num_bits,e_bits=e_bits)
                model.add_module(qname,qlayer)

    #分支汇总
    prefix = blk_name+'_'
    for j in range(len(blk_cfg_table[-1])):
        cfg = blk_cfg_table[-1][j]
        if cfg[0] == 'AD':
            if cfg[1] or ch[0]: #无条件加或flag为true
                qname = 'q_'+prefix+'add%d'%j
                qlayer = QElementwiseAdd(quant_type,2,qi_array=False,qo=True,num_bits=num_bits,e_bits=e_bits)
                model.add_module(qname,qlayer)
        elif cfg[0] == 'RL':
            qname = 'q_'+prefix+'relu%d'%j
            qlayer = QReLU(quant_type,num_bits=num_bits,e_bits=e_bits)
            model.add_module(qname,qlayer)


def ml_utils(model,blk_type,ml_idx,blocks,func,x=None,qo=None):
    ml_name = 'ml%d'%ml_idx
    if blk_type == 'BBLK':
        blk_ch_table = bblk_ch_table
        blk_cfg_table = bblk_cfg_table
    elif blk_type == 'BTNK':
        blk_ch_table = btnk_ch_table
        blk_cfg_table = btnk_cfg_table
    elif blk_type == 'IRES':
        blk_ch_table = ires_ch_table
        blk_cfg_table = ires_cfg_table
    else:
        raise ValueError("ml_quantize: Illegal blk_type")

    last_qo = qo
    x,last_qo = blk_utils(model,blk_ch_table,blk_cfg_table,ml_name,2*ml_idx,0,func,x,last_qo)
    for i in range(1,blocks):
        x,last_qo = blk_utils(model,blk_ch_table,blk_cfg_table,ml_name,2*ml_idx+1,i,func,x,last_qo)
    
    return x, last_qo

def blk_utils(model,blk_ch_table,blk_cfg_table,ml_name,ch_idx,blk_idx,func,x=None,qo=None):
    blk_name = ml_name+'_blk%d'%blk_idx
    ch = blk_ch_table[ch_idx]
    outs = []
    qos = []
    for i in range(2):
        tmp=x
        last_qo = qo
        if i == 0:
            qprefix = 'q_'+blk_name+'_ma_'
        elif i == 1: 
            if ch[0]: #downsample/identity_flag
                qprefix = 'q_'+blk_name+'_ds_'
            else:
                outs.append(tmp)
                qos.append(last_qo)
                continue
       
        for j in range(len(blk_cfg_table[i])):
            cfg = blk_cfg_table[i][j]
            if cfg[0] == 'C':
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
    
    #分支汇总
    qprefix = 'q_'+blk_name+'_'
    for j in range(len(blk_cfg_table[-1])):
        cfg = blk_cfg_table[-1][j]
        if cfg[0] == 'AD':
            if cfg[1] or ch[0]: #无条件加或flag为true
                qname = qprefix+'add%d'%j
                qlayer = getattr(model,qname)
                out = None
                if func == 'forward':
                    out = qlayer(outs)
                elif func == 'inference':
                    out = qlayer.quantize_inference(outs)
                elif func == 'freeze':
                    qlayer.freeze(qos)
                last_qo = qlayer.qo
            else:
                out = outs[0]
                last_qo = qos[0]
        elif cfg[0] == 'RL':
            qname = qprefix+'relu%d'%j
            qlayer = getattr(model,qname)
            if func == 'forward':
                out = qlayer(out)
            elif func == 'inference':
                out = qlayer.quantize_inference(out)
            elif func == 'freeze':
                qlayer.freeze(last_qo)

    return out,last_qo