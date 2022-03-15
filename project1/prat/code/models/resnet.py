import torch 
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64             
        self.C1 = 16                     # C_1 = 64
        self.k = k                       # widening factor

        
        self.conv1 = nn.Conv2d(3, self.C1, kernel_size=3, stride=1,
                            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.C1)
        self.layer1 = self._make_layer(block, self.C1*self.k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*self.C1*self.k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*self.C1*self.k, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 8*self.C1, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out