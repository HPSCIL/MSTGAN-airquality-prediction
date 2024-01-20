import torch
import torch.nn as nn
import torch.nn.functional as F

class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.randn(num_of_timesteps).to(device))
        self.W2 = nn.Parameter(torch.randn(in_channels, num_of_timesteps).to(device))
        self.W3 = nn.Parameter(torch.randn(in_channels).to(device))
        self.bs = nn.Parameter(torch.randn(1, num_of_vertices, num_of_vertices).to(device))
        self.Vs = nn.Parameter(torch.randn(num_of_vertices, num_of_vertices).to(device))


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized
