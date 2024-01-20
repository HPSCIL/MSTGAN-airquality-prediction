import torch
import torch.nn as nn
from MSTAN.code.model.components.Bilinear_Temporal_Attention import Temporal_Attention_layer
from MSTAN.code.model.components.Bilinear_Spatial_Attention import Spatial_Attention_layer


class GSTDM(nn.Module):
    def __init__(self,device, in_channels, num_nodes, num_of_timesteps):
        super(GSTDM,self).__init__()
        self.bilinear_temporal_attention =Temporal_Attention_layer(device, in_channels, num_nodes, num_of_timesteps)
        self.bilinear_spation_attention = Spatial_Attention_layer(device, in_channels, num_nodes, num_of_timesteps)

    def forward(self,x):
        # x:[B,N,F,T]
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        T_attention = self.bilinear_temporal_attention(x)  # [B,T,T]
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), T_attention).reshape(batch_size, num_of_vertices, num_of_features,num_of_timesteps)
        # spatio-temporal dependence weight
        STDG = self.bilinear_spation_attention(x_TAt)  # [B,N,N]
        return STDG








