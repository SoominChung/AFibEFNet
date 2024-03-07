import torch
import torch.nn as nn
from torch.autograd import Variable
import math

## Code based on below links
## https://github.com/gilbutITbook/080289/blob/6ccdc7c0368b96ad9d4637831589041e5a905430/chap07/python_7%EC%9E%A5.ipynb#L1284


class LSTMCell(nn.Module) :
    def __init__(self, input_size, hidden_size, bias=True) :
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=bias)
        self.reset_parameters()
        
    def reset_parameters(self) :
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters() :
            w.data.uniform_(-std, std)
            
    def forward(self, x, hidden) :
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        
        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))
        
        return (hy, cy)

class LSTMModel(nn.Module) :
   def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,device,feature_num=0):
       super(LSTMModel, self).__init__()
       self.hidden_dim = hidden_dim
       
       self.layer_dim = layer_dim
       self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)
       self.fc = nn.Linear(hidden_dim+feature_num, output_dim)
       
       self.fc_feature = nn.Linear(feature_num,feature_num)   
       self.device = device

   def forward(self, x, feature=None) :
       if torch.cuda.is_available() :
           h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device))
       else :
           h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

       if torch.cuda.is_available() :
           c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device))
       else :
           c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

       outs = []
       cn =  c0[0,:,:]
       hn = h0[0,:,:]
       
       for seq in range(x.size(1)) :
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            outs.append(hn)
       out = outs[-1].squeeze()

       if feature != None:
           feature = self.fc_feature(feature)
           out = torch.cat((out,feature),dim=1)

       out = self.fc(out)
       out = torch.sigmoid(out)
       out = out.squeeze()
       return out        