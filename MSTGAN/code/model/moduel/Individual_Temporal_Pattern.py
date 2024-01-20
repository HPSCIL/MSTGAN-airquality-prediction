import torch.nn as nn
import torch.nn.functional as F
from MSTAN.code.model.components.embed import DataEmbedding
from MSTAN.code.model.components.transformer import Transform

class ITPM(nn.Module):
    def __init__(self,in_channels,d_model,out_channels):
        super(ITPM,self).__init__()
        self.enc_embedding = DataEmbedding(in_channels, d_model)
        self.transformer = Transform(d_model,out_channels)
        self.Linear = nn.Conv2d(d_model,out_channels,kernel_size=(1,1))
    def forward(self,x):
        # position embedding
        x_embeding = self.enc_embedding(x.permute(0, 1, 3, 2))
        # multi-station transformer
        x_transformer = self.transformer(x_embeding)
        x_transformer_conv = F.relu(self.Linear(x_transformer.permute(0, 3, 2, 1)).permute(0, 3, 1, 2))
        return x_transformer_conv