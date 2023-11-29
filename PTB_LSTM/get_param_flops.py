from model import *
from functools import partial
from lstm_utils import *

import sys
import torch
from ptflops import get_model_complexity_info

data_path = '../data/ptb'
embed_size = 700
hidden_size = 700
eval_batch_size = 10
dropout = 0.65
tied = True

def lstm_constructor(shape,hidden):
    return {"x":        torch.zeros(shape,dtype=torch.int64),
            "hidden":   hidden}

if __name__ == "__main__":

    corpus = Corpus(data_path)
    ntokens = len(corpus.dictionary)

    model = Model(ntokens, embed_size, hidden_size, dropout, tied)

    full_file = 'ckpt/ptb_PTB_LSTM.pt'
    model.load_state_dict(torch.load(full_file))
    hidden = model.init_hidden(eval_batch_size)
    flops, params = get_model_complexity_info(model, (35,10), as_strings=True, 
                                            input_constructor = partial(lstm_constructor,hidden=hidden),
                                            print_per_layer_stat=True)
