import numpy as np
import matplotlib.pyplot as plt
import sklearn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable


class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTM, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.u_size      = input_size + hidden_size # concat x & h size
 
        # lstm weights
        self.weight_fx = nn.Linear(self.u_size, hidden_size)
        self.weight_ix = nn.Linear(self.u_size, hidden_size)
        self.weight_cx = nn.Linear(self.u_size, hidden_size)
        self.weight_ox = nn.Linear(self.u_size, hidden_size)

    def forward(self,x,state):
        h0, c0 = state   # retrieve previous cell & hidden states

        u = torch.cat((x,h0),1)  # Concatenation of input & previous state 

        f = F.sigmoid(self.weight_fx(u))  # forget gate
        i = F.sigmoid(self.weight_ix(u))  # input gate
        o = F.sigmoid(self.weight_ox(u))  # output gate
        g = F.tanh(self.weight_cx(u))     # candidate cell, c_tilda

        # Compute current states
        cx  = f * c0  +  i * g            # cell state
        hx  = o * F.tanh(cx)              # hidden state

        return hx, cx


