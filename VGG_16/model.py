import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *
import module

# cfg = {
#     'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
#     'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
#     'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
#     'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
# }
feature_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M']
classifier_cfg = [4096, 4096, 'LF']

def make_feature_layers(cfg, batch_norm=True):
    layers = []
    names = []
    input_channel = 3
    idx = 0
    for l in cfg:
        if l == 'M':
            names.append('pool%d'%idx)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        idx += 1
        names.append('conv%d'%idx)
        layers.append(nn.Conv2d(input_channel, l, kernel_size=3, padding=1))

        if batch_norm:
            names.append('bn%d'%idx)
            layers.append(nn.BatchNorm2d(l))

        names.append('relu%d'%idx)
        layers.append(nn.ReLU(inplace=True))
        input_channel = l

    return names,layers

def make_classifier_layers(cfg, in_features, num_classes):
    layers=[]
    names=[]
    idx = 0
    for l in cfg:
        idx += 1
        if l=='LF': #last fc
            names.append('fc%d'%idx)
            layers.append(nn.Linear(in_features,num_classes))
            continue
        names.append('fc%d'%idx)
        layers.append(nn.Linear(in_features,l))
        in_features=l
        
        names.append('crelu%d'%idx)  # classifier relu
        layers.append(nn.ReLU(inplace=True))
        
        names.append('drop%d'%idx)
        layers.append(nn.Dropout())

    return names,layers

def quantize_feature_layers(model,name_list,quant_type,num_bits,e_bits):
    layers=[]
    names=[]
    last_conv = None
    last_bn = None
    idx = 0
    for name in name_list:
        if 'pool' in name:
            names.append('qpool%d'%idx)
            layers.append(QMaxPooling2d(quant_type, kernel_size=2, stride=2, padding=0, num_bits=num_bits, e_bits=e_bits))
        elif 'conv' in name:
            last_conv = getattr(model,name)
        elif 'bn' in name:
            last_bn = getattr(model,name)
        elif 'relu' in name:
            idx += 1
            names.append('qconv%d'%idx)
            if idx == 1:
                layers.append(QConvBNReLU(quant_type, last_conv, last_bn, qi=True, qo=True, num_bits=num_bits, e_bits=e_bits))
            else:
                layers.append(QConvBNReLU(quant_type, last_conv, last_bn, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits))

    return names,layers

def quantize_classifier_layers(model,name_list,quant_type,num_bits,e_bits):
    layers=[]
    names=[]
    idx=0
    for name in name_list:
        layer = getattr(model,name)
        if 'fc' in name:
            idx+=1
            names.append('qfc%d'%idx)
            layers.append(QLinear(quant_type, layer, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits))
        elif 'crelu' in name:
            names.append('qcrelu%d'%idx)
            layers.append(QReLU(quant_type, num_bits=num_bits, e_bits=e_bits))
        elif 'drop' in name:
            names.append(name)
            layers.append(layer)
    
    return names, layers

# 原层和量化层的forward都可以使用该函数
def model_utils(model,feature_name,classifier_name,func, x=None):
    if func == 'inference':
        layer=getattr(model,feature_name[0])
        x = layer.qi.quantize_tensor(x)
    
    last_qo = None
    for name in feature_name:
        layer = getattr(model,name)
        if func == 'forward':
            x = layer(x)
        elif func == 'inference':
            x = layer.quantize_inference(x)
        else: #freeze
            layer.freeze(last_qo)
            if 'conv' in name:
                last_qo = layer.qo
    
    if func != 'freeze':
        x = torch.flatten(x, start_dim=1)
    
    for name in classifier_name:
        layer = getattr(model,name)
        if func == 'forward':
            x = layer(x)
        elif 'drop' not in name:
            if func == 'inference':
                x = layer.quantize_inference(x)
            else: # freeze
                layer.freeze(last_qo)
            
            if 'fc' in name:
                last_qo = layer.qo

    if func == 'inference':
        x = last_qo.dequantize_tensor(x)
    
    return x

class VGG_16(nn.Module):

    def __init__(self, num_class=10):
        super().__init__()
        feature_name,feature_layer = make_feature_layers(feature_cfg,batch_norm=True)
        self.feature_name = feature_name
        for name,layer in zip(feature_name,feature_layer):
            self.add_module(name,layer)
        
        classifier_name,classifier_layer = make_classifier_layers(classifier_cfg,512,num_class)
        self.classifier_name = classifier_name
        for name,layer in zip(classifier_name,classifier_layer):
            self.add_module(name,layer)
        
        # self.fc1 = nn.Linear(512, 4096)
        # self.crelu1 = nn.ReLU(inplace=True)
        # self.drop1 = nn.Dropout()
        # self.fc2 = nn.Linear(4096, 4096)
        # self.crelu2 = nn.ReLU(inplace=True)
        # self.drop2 = nn.Dropout()
        # self.fc3 = nn.Linear(4096, num_class)

    def forward(self, x):
        x = model_utils(self, self.feature_name, self.classifier_name,
                           func='forward', x=x)   
  
        return x

    def quantize(self, quant_type, num_bits=8, e_bits=3):
        # feature
        qfeature_name,qfeature_layer = quantize_feature_layers(self,self.feature_name,quant_type,num_bits,e_bits)
        self.qfeature_name = qfeature_name
        for name,layer in zip(qfeature_name,qfeature_layer):
            self.add_module(name,layer)
        
        # classifier
        qclassifier_name,qclassifier_layer = quantize_classifier_layers(self,self.classifier_name,quant_type,num_bits,e_bits)
        self.qclassifier_name = qclassifier_name
        for name,layer in zip(qclassifier_name,qclassifier_layer):
            if 'drop' not in name:
                self.add_module(name,layer)

        # self.qfc1 = QLinear(quant_type, self.fc1, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        # self.qcrelu1 = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
        # self.qfc2 = QLinear(quant_type, self.fc2, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        # self.qcrelu2 = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
        # self.qfc3 = QLinear(quant_type, self.fc3, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)

    def quantize_forward(self,x):
        x = model_utils(self, self.qfeature_name, self.qclassifier_name,
                           func='forward', x=x)

        return x
    
    def freeze(self):
        model_utils(self, self.qfeature_name, self.qclassifier_name,
                       func='freeze', x=None)
    
    def quantize_inference(self,x):
        x = model_utils(self, self.qfeature_name, self.qclassifier_name,
                           func='inference', x=x)

        return x