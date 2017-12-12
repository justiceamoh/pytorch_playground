import numpy as np
import matplotlib.pyplot as plt
import sklearn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable


class myGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(myGRUCell, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.u_size      = input_size + hidden_size # concat x & h size
 
        # GRU weights
        self.weight_zx = nn.Linear(self.u_size, hidden_size) 
        self.weight_rx = nn.Linear(self.u_size, hidden_size)
        self.weight_hx = nn.Linear(self.u_size, hidden_size)


    def forward(self,x,state):
        u = torch.cat((x,state),1)  # Concatenation of input & previous state 

        z = F.sigmoid(self.weight_zx(u))  # update gate
        r = F.sigmoid(self.weight_rx(u))  # reset gate

        g = F.tanh(self.weight_hx(torch.cat((x,r*state),1))) # candidate cell state
        h = (1 - z)*state + z*g

        return h


