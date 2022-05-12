from zmq import device
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import psycopg2 as pg
import pandas as pd
import sys
import os
from sklearn.metrics import r2_score as sk_r2_score
from train_informer import create_shifted_mean



def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def r2_score_dim(output, target):
    target_mean = torch.mean(target, dim=1, keepdim=True)
    ss_tot = torch.sum((target - target_mean) ** 2, dim=1)
    ss_res = torch.sum((target - output) ** 2, dim=1)
    r2 = 1 - (ss_res / (ss_tot + 1e-5))
    return torch.mean(r2)


class Attention(nn.Module):
    def __init__(self, Output_size, n_class, hidden_size):
        super(Attention, self).__init__()
        self.Output_size = Output_size
        self.n_hidden = hidden_size
        self.n_class = n_class
        self.enc_cell = nn.RNN(
            input_size=self.n_class, hidden_size=self.n_hidden, dropout=0.5
        )
        self.dec_cell = nn.RNN(
            input_size=self.n_class, hidden_size=self.n_hidden, dropout=0.5
        )
        self.attn = nn.Linear(self.n_hidden, self.n_hidden)
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
        attention_weights = torch.nn.functional.softmax(
            torch.bmm(dec_outputs, enc_outputs.transpose(1, 2)), dim=-1
        )
        context = torch.bmm(attention_weights, enc_outputs)
        combine = torch.cat((dec_outputs, context), 2)
        combine = combine.view(combine.shape[0], self.Output_size, -1)
        if self.pls == None:
            self.pls = nn.Linear(combine.shape[2], 1)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.pls = self.pls.to(device)
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


def LSTM(
    training,
    testing,
    batch_size=32,
    Epoch=32,
    n_hidden=128,
    n_class=2,
    learningRate=0.001,
    Output_size=10,
    num_layers=1,
    criterion=nn.MSELoss(),
):
    train, target = training
    test, target_test = testing
    trainer = torch.from_numpy(train).float()
    targeter = torch.from_numpy(target).float()
    targeter_fix = torch.from_numpy(np.array([[i[0] for i in x] for x in targeter])).float()
    trainer_test = torch.from_numpy(test).float()
    targeter_test = torch.from_numpy(target_test).float()
    targeter_test_fix = torch.from_numpy(np.array([[i[0] for i in x] for x in targeter_test])).float()
    dataset = torch.utils.data.TensorDataset(trainer, targeter_fix)
    dtloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    dataset_test = torch.utils.data.TensorDataset(trainer_test,targeter_test_fix)
    dtloader_test = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size, shuffle=False, drop_last=True)
    print("Gpu status",torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden = torch.zeros(num_layers, batch_size, n_hidden)
    hidden = hidden.to(device)
    model = Attention(Output_size, n_class, n_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    # Train
    model.train()
    model.to(device)
    y_lst = []
    output_lst = []
    for epoch in range(Epoch):
        for x, y in dtloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if n_class != 1:
                x = x.squeeze(-1)
                y = y.squeeze(-1)
            else:
                x = x
                y = y
            output, _ = model(x, hidden, x)
            y = y.unsqueeze(-1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()


            r2score = r2_score(output, y)

        if (epoch + 1) % 1 == 0:
            print("Epoch:", "%04d" % (epoch + 1), "MSE =", "{:.6f}".format(loss))
            print("Epoch:", "%04d" % (epoch + 1), "R2 =", "{:.6f}".format(r2score))
            print(
                "Memory Usage:",
                "{:.2f}".format(torch.cuda.memory_allocated(device=device) / 1024**3),
                "GB",
            )
            print(
                "Memory of max",
                "{:.2f}".format(
                    torch.cuda.max_memory_allocated(device=device) / 1024**3
                ),
                "GB",
            )
            print(
                "________________________________________________________________________________________________"
            )
    print("Model has finished training")
    print("Testing...")



    # test the mode using the test set
    model.eval()
    # model.no_grad()
    with torch.no_grad():
        R2_Scores = []
        MAE_Scores = []
        MSE_Scores = []
        R2_1_Scores = []
        plots = []
        trues = []
        preds = []
        print("count", len(dtloader_test))
        for x, y in dtloader_test:
            x = x.to(device)
            y = y.to(device)
            if n_class != 1:
                x = x.squeeze(-1)
                y = y.squeeze(-1)
            else:
                x = x
                y = y
            #print("x shape:", x.shape)
            # print("y shape:", y.cpu().squeeze(-1).shape)
            output, _ = model(x, hidden, x)
            # print("Output shape", output.squeeze(-1).cpu().squeeze(-1).shape)
            # print("X shape", x.cpu().numpy().shape)
            # print("Y shape", y.cpu().unsqueeze(-1).numpy().shape)
            # print("y_hat shape", output.squeeze(-1).cpu().numpy().shape)
            if Output_size == 1 and n_class !=1:
                plots.append((x.cpu(), (y.cpu().unsqueeze(-1), output.squeeze(-1).cpu())))  # (actual, (y, y_hat)) (32,1)
            else:
                plots.append((x.cpu(), (y.cpu(), output.squeeze(-1).cpu())))
            trues.append(y.squeeze(-1).detach().cpu().numpy())
            preds.append(output.squeeze(-1).squeeze(-1).detach().cpu().numpy())
            MSE_Scores.append(criterion(output.cpu(), y.unsqueeze(-1).cpu()))
            MAE_Scores.append(
                MAE(output.detach().cpu().numpy(), y.unsqueeze(-1).detach().cpu().numpy())
            )

            y_lst.append(y.squeeze(-1).cpu().numpy())
            output_lst.append(output.squeeze(-1).cpu().numpy())
            
            R2_Scores.append(r2_score_dim(output.cpu(), y.unsqueeze(-1).cpu()))

        trues = np.array(trues)
        preds = np.array(preds)
        # print("true", trues.shape)
        # print("preds", preds.shape)

        if (Output_size == 1):
            R2_1_Scores.append(sk_r2_score(trues.reshape(-1),preds.reshape(-1)))
    
    preds = create_shifted_mean(np.array(output_lst).tolist(), 1, True)
    trues = create_shifted_mean(np.array(y_lst).tolist(), 1, True)

    r2 = sk_r2_score(trues, preds)

    print("Testing finished")
    print("R2 score:", np.mean([x.item() for x in R2_Scores]))
    print("MSE score:", np.mean([x.item() for x in MSE_Scores]))
    print("MAE score:", np.mean([x.item() for x in MAE_Scores]))



    return (
        model,
        np.mean(r2),
        np.mean([x.item() for x in MSE_Scores]),
        np.mean([x.item() for x in MAE_Scores]),
        plots,
        np.mean(R2_1_Scores) if Output_size == 1 else np.mean([x.item() for x in R2_Scores]),
    )

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def create_shifted_mean(data, stride):
    result = []
    iter = 0
    for i in range(0, len(data)):
        result[i+iter].append(data[i])
        iter += stride
    return [np.mean(x) for x in result]
