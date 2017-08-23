
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class MyModel(nn.Module):
    def __init__(self, pretrained_model):
        super(MyModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.conv = nn.Conv2d(256, 256, kernel_size = 3, padding = 1) 
        self.linear = nn.Linear(256,10) # create layer

    def forward(self, x):
        out = self.pretrained_model(x)
        out = self.conv(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def Finetune(pretrained_model):
    return MyModel(pretrained_model)
