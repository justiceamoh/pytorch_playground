# Stateless GRU
class StatelessNet(nn.Module):
    def __init__(self):
        super(Network, self).__init__() 
        self.gru1 = GRUCell(fbins,L1)
        self.gru2 = GRUCell(L1,L2)
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




#==========================================================
#==========================================================

# Stateful GRU
class StatefulNet(nn.Module):
    def __init__(self):
        super(Network2, self).__init__() 
        self.gru1 = nn.GRUCell(fbins,L1)
        self.gru2 = nn.GRUCell(L1,L2)
        self.fc3  = nn.Linear(L2,L3)
        self.fc4  = nn.Linear(L3,nclass)
        self.h1   = Variable(th.zeros(1, L1))
        self.h2   = Variable(th.zeros(1, L2))
              
    def forward(self,inputs):
        h1 = self.h1
        h2 = self.h2
        for t in range(inputs.size(1)):
            x  = inputs[:,t,:]
            h1 = self.gru1(x,h1)
            h2 = self.gru2(h1,h2)
        
        self.h1 = h1
        self.h2 = h2
        
        ofc3 = F.relu(self.fc3(self.h2))
        out = self.fc4(ofc3)
        return out
    
    def init_hidden(self):
        self.h1 = Variable(th.zeros(1, L1))
        self.h2 = Variable(th.zeros(1, L2))
        return

# Need to use "loss.backward(retain_graph=True)"