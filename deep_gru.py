import numpy as np
import matplotlib.pyplot as plt
import sklearn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable

# # Loaded Spoken Digits Dataset
# dbfile ='../vae/SpokenDigitDB.pkl.gz'
# with gzip.open(dbfile, 'rb') as ifile:
#     df = pd.read_pickle(ifile)
#     print('File loaded as '+ dbfile)

# # Padding & Truncating
# maxlen = 84
# pad    = lambda a, n: a[:,0: n] if a.shape[1] > n else np.hstack((a, np.min(a[:])*np.ones([a.shape[0],n - a.shape[1]])))
# df.Magnitude = df.Magnitude.apply(pad,args=(maxlen,))  # MaxLen Truncation Voodoo :D
# print(np.unique([np.shape(x)[1] for x in df.Magnitude]))

# # Prepare Data
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split

# # Train Scaler
# x_data = df.Magnitude.values
# normsc = np.hstack(x_data)
# scaler = MinMaxScaler().fit(normsc.T)

# # Transform Data using Scaler
# x_data = [scaler.transform(arr.T).T for arr in df.Magnitude.values]
# x_data = np.dstack(x_data).transpose(2,0,1)

# # Add Singleton
# y_data = pd.get_dummies(df.Class).values


# # Shuffle & Split
# x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,
#                               test_size=0.33, random_state=32)

# # Print Dimensions
# print 'Training Feature size:', x_train.shape
# print 'Training Target  size:', y_train.shape
# print ''
# print 'Testing  Feature size:', x_test.shape
# print 'Testing  Target  size:', y_test.shape

    

# Create Torch DataLoader
feats = torch.from_numpy(x_train)
targs = torch.from_numpy(y_train)

dtrain = data_utils.TensorDataset(features, targets)
loader = data_utils.DataLoader(train,batch_size=10,shuffle=True)

# Input Dimensions
_,fbins,steps = x_data.shape
nclass = len(np.unique(y_data))

# Parameters
L1 = 32
L2 = 20
L3 = 16


########################################
# Deep RNN Declaration 0 - Sequential  #
########################################
model = torch.nn.Sequential(
    nn.GRU(fbins,L1,1),
    nn.GRU(L1,L2,1),
    nn.GRU(L2,L3,1),
    nn.Linear(L3,nclass)
)

model.double()


########################################
# Deep RNN Declaration 1 - Layer Level #
########################################
# Create Torch DataLoader
feats = torch.from_numpy(x_train)
targs = torch.from_numpy(y_train)

dtrain = data_utils.TensorDataset(features, targets)
loader = data_utils.DataLoader(train,batch_size=10,shuffle=True)

# Using GRU layers, Without Time Loop
gru1 = nn.GRU(fbins,L1).double()
gru2 = nn.GRU(L1,L2).double()
fc3  = nn.Linear(L2,L3).double()
fc4  = nn.Linear(L3,nclass).double()

for i, data in enumerate(loader):
    outputs = []
    x,y = data
    x,y = Variable(x.permute(2,0,1)),Variable(y)
    
    h1  = Variable(torch.zeros(1,x.size(1), L1)).double()
    h2  = Variable(torch.zeros(1,x.size(1), L2)).double()

    o1,h1 = gru1(x,h1)     # return output sequence o1
    o2,h2 = gru2(o1,h2)    # return output sequence o2
    lin   = F.relu(fc3(h2)) # use last state
    out   = F.softmax(fc4(lin),dim=0)



#######################################
# Deep RNN Declaration 2 - Cell Level #
#######################################
# Using Cell, With Time Loop 
gru1 = nn.GRUCell(fbins,L1).double()
gru2 = nn.GRUCell(L1,L2).double()
fc3  = nn.Linear(L2,L3).double()
fc4  = nn.Linear(L3,nclass).double()

ht1 = Variable(torch.zeros(1, L1)).double()
ht2 = Variable(torch.zeros(1, L2)).double()

for i, data in enumerate(loader):
    outputs = []
    x,y = data
    x,y = Variable(x.permute(2,0,1)),Variable(y)
    
    for xt1 in x:
        ht1 = gru1(xt1,ht1)
        ht2 = gru2(ht1,ht2)
        
    ot3 = F.relu(fc3(ht2))
    out = fc4(ot3)
