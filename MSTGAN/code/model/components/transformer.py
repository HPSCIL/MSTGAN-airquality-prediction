import torch
import torch.nn as nn
from torch.autograd import Variable
import math
# from visualizer import get_local

class Transform(nn.Module):
    def __init__(self, outfea, d):
        super(Transform, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )
        self.d = d
    # @get_local('A')
    def forward(self, x):
        # q,k,v linear
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)

        # multi-head attention
        query = torch.cat(torch.split(query, self.d, -1), 0)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,1,3,2)
        value = torch.cat(torch.split(value, self.d, -1), 0)

        A = torch.matmul(query, key)
        A /= (self.d ** 0.5)
        A = torch.softmax(A,-1)

        value = torch.matmul(A, value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1)
        value += x

        # Layer normalization and feed forward
        value = self.ln(value)
        x = self.ff(value)
        output = self.lnff(x)
        return output

