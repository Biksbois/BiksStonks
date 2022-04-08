import torch
import torch.nn as nn
import torch.nn.functional as F

class Iwata_simple(nn.Module):
   """Model as from Iwata Paper:
      - Bidirectional LSTM to encode support set
      - Forward LSTM to encode query
      - Attention mechansim to relate query representation to support set
      - Feed the output of attention and query lstm into ffn to give output X^t+1
      Note: only supports N_q = 1 (query set size)"""
   def __init__(self, enc_in, hidden_size, c_out, s_n_layers, bidirectional=True, 
                dropout=0.0, device=torch.device('cuda:0')):
      super(Iwata_simple, self).__init__()
      assert hidden_size % 8 == 0 # number of att. heads
      self.support_encoder = nn.LSTM(input_size=enc_in, hidden_size=hidden_size//2, 
                                       num_layers=s_n_layers, dropout=dropout, 
                                       bidirectional=bidirectional)
      
      self.query_encoder = nn.LSTM(input_size=enc_in, hidden_size=hidden_size, dropout=dropout)
      self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
      # g is a feed forward network - as from paper
      self.g = nn.Sequential(nn.Linear(hidden_size*2, hidden_size), nn.ReLU(),
                             nn.Linear(hidden_size, c_out))

   def forward(self, support_set, query_set):
      """
      Args: 
         support_set: (batch_size, seq_len, enc_in), batch_size = support_size
         query_set: (1, seq_len, enc_in)
      Returns:
      Note: only supports N_q = 1 (query set size) """
      assert query_set.shape[0] == 1
      # Encode support set
      support_set = support_set.permute(1, 0, 2)
      support_enc_out, _ = self.support_encoder(support_set)
      # Encode query set
      query_set = query_set.permute(1, 0, 2)
      query_enc_out, (z, c) = self.query_encoder(query_set)
      # Attention 
      # broadcast query_enc_out from 
      # (seq_len, 1, hidden_size) -> (seq_len, batch_size, hidden_size) 
      query_enc_out = query_enc_out.repeat(1, support_set.shape[1], 1) 
      a = self.attention(query_enc_out, support_enc_out, support_enc_out)[0] 
      # sum over support set and time steps to get a as from paper 
      a = a.sum(dim=(0,1)) 
      # a,z (64*2)        
      z = z.squeeze(0).squeeze(0) 
      return self.g(torch.cat((a, z))) 