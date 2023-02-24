import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Net(nn.Module):
  def __init__(self,dropout=0.0):
    super().__init__()
    self.prep = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                              nn.BatchNorm2d(16),
                              nn.ReLU(),
                              nn.Dropout2d(dropout),
                              nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                              nn.BatchNorm2d(32),
                              nn.ReLU(),
                              nn.Dropout2d(dropout),
                              nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                              nn.BatchNorm2d(48),
                              nn.ReLU(),
                              nn.Dropout2d(dropout)
                              )
    
    self.GAP = nn.AvgPool2d(32,32) 
    
    self.fc_uin =   nn.Linear(48,8, bias=False) 
    self.fc_uout =  nn.Linear(8,48, bias=False) 

    self.fc_output =   nn.Linear(48,10, bias=False)

    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    x = self.prep(x)    
    x = self.GAP(x)    
    x = x.view(x.size(0), -1)

    y = x

    # ULTIMUS 
    for i in range(0,4):
        Q = self.fc_uin(x)
        V = self.fc_uin(x)
        K = self.fc_uin(x)

        mult = torch.matmul(torch.transpose(Q, 0, 1), K) / 8 ** 0.5

        AM = F.softmax(mult, dim=1)
        
        Z = torch.matmul(V, AM)        
        
        x_out = self.fc_uout(Z)

        x = F.relu(x_out) + y
        x = self.dropout(x)

    x = F.relu(x)
    x = self.fc_output(x)    
    
    return x
