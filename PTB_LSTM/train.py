# coding: utf-8
import time
import math
import os
import torch
import torch.nn as nn
from lstm_utils import *
from model import *
import sys

data_path = '../data/ptb'
embed_size = 700
hidden_size = 700
lr = 22
clip = 0.25
epochs = 10
train_batch_size = 20
eval_batch_size = 10
bptt = 35 #所取的串长度
dropout = 0.65
tied = True
seed = 1111
seed_gpu = 1111
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_interval = 200
save_path = 'ckpt/ptb_PTB_LSTM.pt'


def evaluate(model, eval_data, ntokens, eval_batch_size, bptt):
    model.eval()
    total_loss = 0.
    lossLayer = nn.CrossEntropyLoss()
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            output, hidden = model(data,hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * lossLayer(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(eval_data) - 1)

def train(model, train_data, ntokens, train_batch_size, bptt, lr, clip):
    model.train()
    total_loss = 0.
    lossLayer = nn.CrossEntropyLoss()
    start_time = time.time()
    hidden = model.init_hidden(train_batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        # hidden = repackage_hidden(hidden)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer.zero_grad()
        output, hidden = model(data,hidden)
        hidden = repackage_hidden(hidden)
        loss = lossLayer(output.view(-1, ntokens), targets)
        loss.backward()

        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(p.grad.data, alpha=-lr)

        total_loss += loss.item()

        if batch % 200 == 0 and batch > 0:
            cur_loss = total_loss / 200
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / 200, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed_gpu)

    corpus = Corpus(data_path)
    ntokens = len(corpus.dictionary)

    train_data = batchify(corpus.train, train_batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    model = Model(ntokens, embed_size, hidden_size, dropout, tied).to(device)

    best_val_loss = None

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(model, train_data, ntokens, train_batch_size, bptt, lr, clip)
        val_loss = evaluate(model, val_data, ntokens, eval_batch_size, bptt)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(save_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            best_val_loss = val_loss
        else:
            lr /= 2.5
            if lr < 0.0001:
                break

    with open(save_path, 'rb') as f:
        # model = torch.load(f)
        model = Model(ntokens, embed_size, hidden_size, dropout, tied).to(device)
        model.load_state_dict(torch.load(f))
        #将LSTM层的参数拉平为一维，方便并行计算
        lstm_flatten(model)
        # model.lstm3.flatten_parameters()
        # model.lstm4.flatten_parameters()

    test_loss = evaluate(model, test_data, ntokens, eval_batch_size, bptt)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

