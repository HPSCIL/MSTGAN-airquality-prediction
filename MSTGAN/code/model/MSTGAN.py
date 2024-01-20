import torch.nn as nn
from MSTAN.code.model.ST_block import MST_block

class MSTAN(nn.Module):
    def __init__(self,input_dim,hiden_dim,out_channels,device,num_nodes,num_of_timesteps,num_for_predict,K,dropout,d_model):
     super(MSTAN,self).__init__()
     self.MST_Blocklist = nn.ModuleList()
     self.MST_Blocklist.append(MST_block(input_dim,hiden_dim,device,num_nodes,num_of_timesteps,K,dropout,d_model))
     self.MST_Blocklist.append(MST_block(hiden_dim,out_channels,device,num_nodes,num_of_timesteps,K,dropout,d_model))
     self.Predict_layer = nn.Conv2d(num_of_timesteps, num_for_predict, kernel_size=(1, out_channels))



    def forward(self,x,cheb_polynomials):
        # x:[B,N,F,T]
        # Multi-Spatio-temporal Feature extraction
        for block in self.MST_Blocklist:
            x= block(x,cheb_polynomials)
        # Prediction layer
        output = self.Predict_layer(x.permute(0, 3, 1, 2)).permute(0,2,3,1)
        return output


