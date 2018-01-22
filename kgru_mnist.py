import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.constraints import UnitNorm
from torchsample.initializers import XavierUniform
from torchsample.metrics import CategoricalAccuracy

import os
from torchvision import datasets
ROOT = '/Users/d30676n/pytorch/mnist'
dataset = datasets.MNIST(ROOT, train=True, download=True)
x_train, y_train = th.load(os.path.join(dataset.root, 'processed/training.pt'))
x_test, y_test = th.load(os.path.join(dataset.root, 'processed/test.pt'))

x_train = x_train.float()
y_train = y_train.long()
x_test  = x_test.float()
y_test  = y_test.long()

x_train = x_train / 255.
x_test  = x_test / 255.

# Use Subset
# only train on a subset
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# Input Dimensions
fbins,steps = 28,28 
nclass = 10 

# Parameters
L1 = 32
L2 = 20
L3 = 16


# kGRU Cell
class kGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(kGRUCell, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.u_size      = input_size + hidden_size # concat x & h size
 
        # GRU weights
        self.weight_zx = nn.Linear(self.u_size, hidden_size) 
        self.weight_hx = nn.Linear(self.u_size, hidden_size)

    def forward(self,x,state):
        u = th.cat((x,state),1)  # Concatenation of input & previous state 
        z = F.sigmoid(self.weight_zx(u))  # update gate
        g = F.tanh(self.weight_hx(u)) # candidate cell state
        h = (1 - z)*state + z*g
        return h


# Define Model - Stateless GRU
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__() 
        self.gru1 = kGRUCell(fbins,L1)
        self.gru2 = kGRUCell(L1,L2)
        self.fc3  = nn.Linear(L2,L3)
        self.fc4  = nn.Linear(L3,nclass)
              
    def forward(self,inputs):
        h1 = Variable(th.zeros(inputs.size(0), L1))
        h2 = Variable(th.zeros(inputs.size(0), L2))
        for t in range(inputs.size(1)):
            x  = inputs[:,t,:]
            h1 = self.gru1(x,h1)
            h2 = self.gru2(h1,h2)            
        ofc3 = F.relu(self.fc3(h2))
        out = self.fc4(ofc3)
        return F.log_softmax(out)


net     = Network()
model   = ModuleTrainer(net)
metrics = [CategoricalAccuracy(top_k=3)]


model.compile(loss='nll_loss',
                optimizer='adadelta',
                metrics=metrics)

model.fit(x_train,y_train,
          val_data=(x_test,y_test),
          num_epoch=10,
          batch_size=128,
          verbose=1)

model.evaluate(x_test,y_test)

