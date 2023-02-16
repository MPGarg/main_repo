import torch
import torch.nn as nn
import torch.nn.functional as F


class Custom_ResNet(nn.Module):
  def __init__(self,dropout=0.0):
    super().__init__()
    
    self.prep = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                              nn.BatchNorm2d(64),
                              nn.ReLU(),
                              nn.Dropout2d(dropout)
                              )
    
    self.layer1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.MaxPool2d(2,2), 
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Dropout2d(dropout) 
                                )
    
    self.resblock1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Dropout2d(dropout),
                                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(), 
                                    nn.Dropout2d(dropout)
                                    )
            
    self.layer2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.MaxPool2d(2,2), 
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Dropout2d(dropout)
                                )
    
    self.layer3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.MaxPool2d(2,2), 
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Dropout2d(dropout)
                                )
    
    self.resblock2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Dropout2d(dropout),
                                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    nn.Dropout2d(dropout) 
                                    )
    
    self.pool = nn.MaxPool2d(4,4) 
    
    self.fc =   nn.Linear(512,10, bias=False) 
    
  def forward(self, x):
    y = self.prep(x)
    
    y = self.layer1(y)
    r1  = self.resblock1(y)
    y = y + r1
    
    y = self.layer2(y)
    
    y = self.layer3(y)
    r2 = self.resblock2(y)
    y = y + r2
    
    y = self.pool(y)
	
    y = y.view(y.size(0), -1)
    y = self.fc(y)
    
    return F.log_softmax(y,dim=1)