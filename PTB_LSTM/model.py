import torch.nn as nn
from module import *

class Model(nn.Module):
    def __init__(self,ntoken, ninp, nhid, dropout=0.5, tie_weights=False):
        super(Model, self).__init__()
        self.embed1 = nn.Embedding(ntoken, ninp)
        self.drop2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(ninp, nhid, 1, dropout=dropout)
        # self.drop4 = nn.Dropout(dropout)
        self.lstm4 = nn.LSTM(nhid, nhid, 1)
        self.drop5 = nn.Dropout(dropout)
        self.fc6 = nn.Linear(nhid, ntoken)

        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('the number of hidden unit per layer must be equal to the embedding size')
            self.fc6.weight = self.embed1.weight

        self.init_weights()

        self.nhid = nhid

    def init_weights(self):
        initrange = 0.1
        self.embed1.weight.data.uniform_(-initrange, initrange)
        self.fc6.bias.data.zero_()
        self.fc6.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, hidden=None):
        x = self.embed1(x)
        x = self.drop2(x)
        x, hidden = self.lstm3(x, hidden)
        x, hidden = self.lstm4(x, hidden)
        x = self.drop5(x)
        t, n = x.size(0), x.size(1)
        x = x.view(t*n,-1)
        x = self.fc6(x)
        x = x.view(t,n,-1)
        return x, hidden

    def init_hidden(self,bsz):
        # 获取模型第一个参数的device和数据类型
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, self.nhid),
                weight.new_zeros(1, bsz, self.nhid))

    def quantize(self, quant_type, num_bits=8, e_bits=3):
        # embed input作为索引，无需量化
        self.qembed1 = QEmbedding(quant_type, self.embed1, num_bits=num_bits, e_bits=e_bits)
        #qix承接上层，无需再量化，qih和qic初次对hidden操作，需量化
        # self.qlstm3 = QLSTM(quant_type, self.lstm3, has_hidden=True, qix=False, qih=True, qic=True, num_bits=num_bits, e_bits=e_bits)
        self.qlstm3 = QLSTM(quant_type, self.lstm3, has_hidden=True, qix=False, qih=True, qic=True, num_bits=num_bits, e_bits=e_bits)
        self.qlstm4 = QLSTM(quant_type, self.lstm4, has_hidden=True, qix=False, qih=False, qic=False, num_bits=num_bits, e_bits=e_bits)
        self.qfc6 = QLinear(quant_type, self.fc6, num_bits=num_bits, e_bits=e_bits)

    def quantize_forward(self, x, hidden=None):
        x = self.qembed1(x)
        x = self.drop2(x)
        x,hidden = self.qlstm3(x,hidden)
        x,hidden = self.qlstm4(x,hidden)
        x = self.drop5(x)
        t,n = x.size(0), x.size(1)
        x = x.view(t*n, -1)
        x = self.qfc6(x)
        x = x.view(t,n,-1)
        return x,hidden

    def freeze(self):
        self.qembed1.freeze()
        self.qlstm3.freeze(qix=self.qembed1.qo)
        self.qlstm4.freeze(qix=self.qlstm3.qox, qih=self.qlstm3.qoh, qic=self.qlstm3.qoc)
        self.qfc6.freeze(qi=self.qlstm4.qox)

    def quantize_inference(self, x, hidden=None):
        x = self.qembed1.quantize_inference(x)
        x,hidden = self.qlstm3.quantize_inference(x,hidden)
        x,hidden = self.qlstm4.quantize_inference(x,hidden)
        t,n = x.size(0), x.size(1)
        x = x.view(t*n, -1)
        x = self.qfc6.quantize_inference(x)
        x = x.view(t,n,-1)
        return x,hidden
        