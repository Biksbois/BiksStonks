
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import psycopg2 as pg
import pandas as pd
import sys
import os

class Attention(nn.Module):
    def __init__(self,Output_size,n_class,hidden_size):
        super(Attention, self).__init__()
        self.enc_cell = nn.RNN(input_size=self.n_class, hidden_size=self.n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(input_size=self.n_class, hidden_size=self.n_hidden, dropout=0.5)
        self.attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.Output_size = Output_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.pls = None
        
    def forward(self, enc_inputs, hidden, dec_inputs):
        dec_inputs = dec_inputs.transpose(0, 1) 
        enc_inputs = enc_inputs.transpose(0, 1) 
        enc_outputs, enc_hidden = self.enc_cell(enc_inputs, hidden)
        trained_attn = []
        hidden = enc_hidden
        n_step = len(dec_inputs)
        dec_outputs, hidden = self.dec_cell(dec_inputs, hidden)
        enc_outputs = enc_outputs.transpose(0, 1)
        dec_outputs = dec_outputs.transpose(0, 1)
        enc_outputs = self.attn(enc_outputs)
        attention_weights = torch.nn.functional.softmax(torch.bmm(dec_outputs, enc_outputs.transpose(1,2)), dim=-1)
        context = torch.bmm(attention_weights, enc_outputs)
        combine = torch.cat((dec_outputs, context), 2)
        combine = combine.view(combine.shape[0], self.Output_size, -1)
        if self.pls == None:
            self.pls = nn.Linear(combine.shape[2], 1)
        return self.pls(combine), trained_attn

    def get_att_weight(self, dec_output, enc_outputs):
        n_step = len(enc_outputs)
        attn_scores = torch.zeros(n_step)

        for i in range(n_step):
            attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])

        return F.softmax(attn_scores).view(1, 1, -1)

    def get_att_score(self, dec_output, enc_output): 
        score = self.attn(enc_output)
    
        return score

def LSTM(data):
    n_hidden = 128 
    n_class = 2
    PointSize = 200
    Epoch = 32
    batch_size = 32
    num_layers = 1
    Output_size = 10
    learningRate = 0.001

    train = np.array([np.array(d[:PointSize-Output_size]) for d in data])
    target = np.array([np.array(d[PointSize-Output_size:]) for d in data])
    trainer = torch.from_numpy(train).float()
    targeter = torch.from_numpy(target).float()
    dataset = torch.utils.data.TensorDataset(trainer,targeter)
    dtloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    hidden = torch.zeros(num_layers, batch_size, n_hidden)
    model = Attention(Output_size,n_class,n_hidden)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learningRate=0.001)   
    # Train
    model.train()
    for epoch in range(Epoch):
        for x, y in dtloader:
            optimizer.zero_grad()
            x = x.squeeze(-1)
            y = y.squeeze(-1)
            output, _ = model(x, hidden, x)
            loss = criterion(output, y.squeeze(0))
        if (epoch + 1) % 5 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'MSE =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
from itertools import islice

def window1(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result