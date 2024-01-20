import torch
import torch.nn as nn
import torch.nn.functional as F

class Temporal_Attention_layer(nn.Module):
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.randn(num_of_vertices).to(device))  #35
        self.U2 = nn.Parameter(torch.randn(in_channels, num_of_vertices).to(device))  #[6,35]
        self.U3 = nn.Parameter(torch.randn(in_channels).to(device))  #[6]
        self.be = nn.Parameter(torch.randn(1, num_of_timesteps, num_of_timesteps).to(device))  #[1,1,1]
        self.Ve = nn.Parameter(torch.randn(num_of_timesteps, num_of_timesteps).to(device))  #[1,1]

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape  #[32,35,6,48]
        lhs = torch.matmul(torch.matmul(x.permute(0,3,2,1), self.U1), self.U2)   #
        # x:(B, N, F_in, T) -> (B, T, F_in, N)  [32,24,6,35]
        # (B, T, F_in, N)(N) -> (B,T,F_in)  [32,24,6]
        # (B,T,F_in)(F_in,N)->(B,T,N)  [32,24,35]

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)   [32,35,24]

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)  [32,24,24]

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)  [32,24,24]

        E_normalized = F.softmax(E, dim=1)   #归一化

        return E_normalized