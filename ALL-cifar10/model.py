import torch.nn as nn

from cfg import *
from module import *
from model_deployment import *


class Model(nn.Module):
    def __init__(self,model_name,num_classes=10):
        super(Model, self).__init__()
        self.cfg_table = model_cfg_table[model_name]
        make_layers(self,self.cfg_table)
        # # 参数初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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

# if __name__ == "__main__":
#     model = Inception_BN()
#     model.quantize('INT',8,3)
#     print(model.named_modules)
#     print('-------')
#     print(model.named_parameters)
#     print(len(model.conv0.named_parameters()))