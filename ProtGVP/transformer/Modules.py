import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) # b x n x lq x lk
        
        if mask is not None:
            # mask   src b x 1 x 1 x l  tar b x 1 x l x l 
            attn = attn.masked_fill(mask == 0, -1e9) # mask是一个矩阵，为0的值表示该位置需要被mask。

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn

# attn   b x n x l x l