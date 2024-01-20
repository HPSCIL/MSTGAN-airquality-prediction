import torch
import torch.nn as nn
from MSTAN.code.model.components.GRU import GRUCell

class STDG_CGRU(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self,device,K,num_nodes,in_channels, out_channels,num_of_timesteps):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(STDG_CGRU, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = out_channels
        self.num_nodes = num_nodes
        self.device =device
        self.Theta = nn.ParameterList([nn.Parameter(torch.randn(in_channels, out_channels).to(self.device)) for _ in range(K)])
        self.GRUCells = GRUCell(device, self.num_nodes, self.hidden_dim)
        self.conv = nn.Conv2d(1, num_of_timesteps, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()

    def forward(self, x,STDG,cheb_polynomials):

        batch_size, num_nodes, in_channels, num_of_timesteps = x.shape
        # initial hiden state h0
        state = torch.zeros(batch_size, num_nodes, self.hidden_dim).to(self.device)
        inner_states = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in,1)
            output = torch.zeros(batch_size, num_nodes, self.out_channels).to(self.device)  # (b, N, F_out)
            # STDG-chebyshev graph convolution
            for k in range(self.K):
                T_k = cheb_polynomials[k]
                # add STDG
                T_k_with_at = T_k.mul(STDG)
                theta_k = self.Theta[k]
                rhs = T_k_with_at.permute(0,2,1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)
            # GRU
            state = self.GRUCells(output, state)
            inner_states.append(state)
        current_inputs = torch.stack(inner_states, dim=-1)
        # output final hiden state ht
        out = self.conv(current_inputs[:,:,:,-1:].permute(0,3,1,2))

        return  out
