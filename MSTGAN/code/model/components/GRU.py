import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUCell(nn.Module):
    def __init__(self, device, node_num, hidden_dim):
        super(GRUCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.Liner1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.Liner2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.device = device

    def forward(self, x, state):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(self.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = self.Liner1(input_and_state)  # input_and_state @ self.Wz_r +self.bias_zr
        z_r = torch.sigmoid(z_r)
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, r * state), dim=-1)
        hc = self.Liner2(candidate)  # candidate @ self.Wc +self.bias_c
        hc = torch.tanh(hc)
        h = z * state + (1 - z) * hc
        return h