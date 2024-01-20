import torch.nn as nn
import torch.nn.functional as F
from MSTAN.code.model.moduel.Individual_Temporal_Pattern import ITPM
from MSTAN.code.model.moduel.Global_ST_model import GSTDM
from MSTAN.code.model.moduel.Local_Feature_model import STDG_CGRU


class MST_block(nn.Module):
    def __init__(self,in_channels,out_channels,device,num_nodes,num_of_timesteps,K,dropout,d_model):
        super(MST_block,self).__init__()
        self.ITPM = ITPM(in_channels,d_model,out_channels)
        self.GSTDM = GSTDM(device,out_channels,num_nodes, num_of_timesteps)
        self.STDG_CGRU = STDG_CGRU(device,K,num_nodes,in_channels, out_channels,num_of_timesteps)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)




    def forward(self,x,cheb_polynomials):
       # x: [B,N,F,T]

       # Individual temporal temporal pattern moduel
       x_ITPM = self.ITPM(x)
       #global spatio-temporal dependence moduel
       STDG = self.GSTDM(x_ITPM)
       #local spatio-temporal feature moduel
       ST_out = self.STDG_CGRU(x,STDG,cheb_polynomials)

       # residual_layer
       x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
       x_residual = self.dropout(self.ln(F.relu(x_residual + ST_out.permute(0,3,2,1)).permute(0, 3, 2, 1)).permute(0, 2, 3, 1))
       return x_residual