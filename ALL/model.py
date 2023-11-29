import torch.nn as nn

from cfg import *
from module import *
from model_deployment import *


class Model(nn.Module):
    def __init__(self,model_name,dataset):
        super(Model, self).__init__()
        self.cfg_table = model_cfg_table[model_name]
        adapt_dataset(self.cfg_table,dataset)
        make_layers(self,self.cfg_table)
        self.model_state = None
        
    def forward(self,x,out_feature=False):
        if self.model_state is None:
            return model_forward(self,self.cfg_table,x,out_feature)
        elif self.model_state == 'quantize':
            return self.quantize_forward(x,out_feature)
        elif self.model_state == 'freeze':
            return self.quantize_inference(x,out_feature)
        else:
            assert False, "Illegal Model State"

    def quantize(self, quant_type, num_bits=8, e_bits=3):
        model_quantize(self,self.cfg_table,quant_type,num_bits,e_bits)
        self.model_state = 'quantize'
    
    def quantize_forward(self,x,out_feature=False):
        return model_utils(self,self.cfg_table,func='forward',x=x,out_feature=out_feature)

    def freeze(self):
        model_utils(self,self.cfg_table,func='freeze')
        self.model_state = 'freeze'
    
    def quantize_inference(self,x,out_feature=False):
        return model_utils(self,self.cfg_table,func='inference',x=x,out_feature=out_feature)

    def fakefreeze(self):
        model_utils(self,self.cfg_table,func='fakefreeze')

    def get_output_layer_weight(self):
        return get_output_layer_weight(self,self.cfg_table)