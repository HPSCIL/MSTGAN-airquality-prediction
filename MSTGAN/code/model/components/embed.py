import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()  #[5000,512]
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  #[5000,1]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()  #[256,]

        pe[:, 0::2] = torch.sin(position * div_term)  #从0开始取，每跳两个取一个值
        pe[:, 1::2] = torch.cos(position * div_term)  #从1开始取，每跳两个取一个值

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):   #[32,35,6,24]
        return self.pe[:, :x.size(2)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=(1,1))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0,3,2,1)).permute(0,3,2,1)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)


    def forward(self, x):
        #x:[B,N,T,F]
        x1 = self.position_embedding(x)
        x = self.value_embedding(x) + x1
        
        return F.relu(x)