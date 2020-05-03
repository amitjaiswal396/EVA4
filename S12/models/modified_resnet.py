import torch.nn as nn
import torch.nn.functional as F

class ModifiedResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ModifiedResBlock, self).__init__()
        self.layerconv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
        )
        ### This layer applies after the first conv and we intend to keep the channel size same
        self.resconv = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
            )
        
    def forward(self, x):
        out = self.layerconv(x)
        res = self.resconv(out)
        return out+res
        
class s11DNN(nn.Module):
    def __init__(self):
        super(s11DNN, self).__init__()

        self.prepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = ModifiedResBlock(64, 128, 1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer3 = ModifiedResBlock(256, 512, 1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, x):
        out = self.prepLayer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out