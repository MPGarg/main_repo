import torch.nn as nn
import torch

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        skip = x  # store the output for the skip connection
        x = self.maxpool(x)
        
        return x, skip

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='transpose'):
        super(ExpandingBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        if mode=='transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                          nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1))

    def forward(self, x, skip):
        
        x = self.upsample(x)
        x = torch.cat((x, skip), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)        

        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.contract1 = ContractingBlock(in_channels, 32)
        self.contract2 = ContractingBlock(32, 64)
        self.contract3 = ContractingBlock(64, 128)
        self.contract4 = ContractingBlock(128, 256)
        self.contract5 = ContractingBlock(256, 512)

        self.expand1 = ExpandingBlock(512, 256)        
        self.expand2 = ExpandingBlock(256, 128)
        self.expand3 = ExpandingBlock(128, 64)
        self.expand4 = ExpandingBlock(64, 32)
        
        self.conv_sec = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(3, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Contracting path
        x, skip1 = self.contract1(x)
        x, skip2 = self.contract2(x)
        x, skip3 = self.contract3(x)
        x, skip4 = self.contract4(x)
        _, x = self.contract5(x)
        
        # Expanding path
        x = self.expand1(x, skip4)
        x = self.expand2(x, skip3)
        x = self.expand3(x, skip2)
        x = self.expand4(x, skip1)

        x = self.relu1(self.conv_sec(x))

        return self.final_conv(x)
     