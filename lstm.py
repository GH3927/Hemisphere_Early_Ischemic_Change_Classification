# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:03:21 2021

@author: gyuha
"""

import torch
from torch import nn
from torch.nn import functional as F

no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
Tensor = torch.cuda.FloatTensor if use_cuda else torch.Tensor

def EfficientNet_series(model, data, sequence=False):  
    with torch.no_grad():
        b, n, c, h, w = data.shape # (b, n, c, h, w)
        features = model(data[0]).unsqueeze(0) # (1, n, f)
        
        for i in range(1,len(data)):
            features = torch.cat((features,model(data[i]).unsqueeze(0)),dim=0) # (b, n, f)
                
        if sequence:
            b, n, f = features.shape
            
            features_lag = torch.zeros((b,n,f)).to(device)
            features_lead = torch.zeros((b,n,f)).to(device)        
    
            features_lag[:,1:,:] = features[:,1:,:] - features[:,:-1,:]
            features_lead[:,:-1,:] = features[:,:-1,:] - features[:,1:,:]
            
            features = torch.cat((features, features_lag, features_lead),dim=-1)
    
    return features
    
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class LSTM(nn.Module):
    def __init__(self, embed_size=1792, LSTM_UNITS=1792, DO = 0.3, drop=False):
        super(LSTM, self).__init__()
        self.drop = drop
        self.embedding_dropout = SpatialDropout(DO)
        self.embed_size = embed_size

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS*2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear3 = nn.Linear(LSTM_UNITS*2, 1)
        
    def forward(self, x):
        if self.drop:
            h_embedding = self.embedding_dropout(x)
        else: h_embedding = x
        
        h_embadd = torch.cat((h_embedding[:,:,:self.embed_size], h_embedding[:,:,:self.embed_size]), -1)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2))
        
        hidden1 = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embadd
        hidden2 = self.linear3(hidden1)
        output = hidden2[:,-1,:]
 
        return output
   
class GRU(nn.Module):
    def __init__(self, embed_size=1792, GRU_UNITS=1792, DO = 0.3, drop=False):
        super(GRU, self).__init__()
        self.drop = drop
        self.embedding_dropout = SpatialDropout(DO)
        self.embed_size = embed_size
        
        self.gru1 = nn.GRU(embed_size, GRU_UNITS, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(GRU_UNITS*2, GRU_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(GRU_UNITS*2, GRU_UNITS*2)
        self.linear2 = nn.Linear(GRU_UNITS*2, GRU_UNITS*2)
        self.linear3 = nn.Linear(GRU_UNITS*2, 1)
        
    def forward(self, x):
        if self.drop:
            h_embedding = self.embedding_dropout(x)
        else: h_embedding = x
        
        h_embadd = torch.cat((h_embedding[:,:,:self.embed_size], h_embedding[:,:,:self.embed_size]), -1)
        
        h_gru1, _ = self.gru1(h_embedding)
        h_gru2, _ = self.gru2(h_gru1)
        
        h_conc_linear1  = F.relu(self.linear1(h_gru1))
        h_conc_linear2  = F.relu(self.linear2(h_gru2))
        
        hidden1 = h_gru1 + h_gru2 + h_conc_linear1 + h_conc_linear2 + h_embadd
        hidden2 = self.linear3(hidden1)
        output = hidden2[:,-1,:]
 
        return output