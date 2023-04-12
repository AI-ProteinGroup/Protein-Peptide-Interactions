import torch
import torch.nn as nn
import numpy as np
import transformer.Models as TFmodels
from se3_transformer_pytorch import SE3Transformer
#SE3Transformer

class MyNet(nn.Module):
    
    def __init__(
            self, 
            se3_dim = 20,
            se3_heads = 8,
            se3_depth = 6,
            se3_dim_head = 64,
            se3_num_degrees = 4,
            se3_valid_radius = 6,
            se3_reduce_dim_out = True,

            n_src_vocab = 500, 
            n_trg_vocab = 23, 
            src_pad_idx=0, 
            trg_pad_idx=0,
            d_word_vec=512,   # word_vec = d_model 
            d_model=512, 
            d_inner=2048,
            n_layers=6, 
            n_head=8, 
            d_k=64, 
            d_v=64, 
            dropout=0.1, 
            n_position_src=2050,
            n_position_trg=200,#位置编码个数
            batch=4,
            trg_pad_len=6,
            trg_emb_prj_weight_sharing=False,  # decoder的embedding和output fully connecting layer共享参数
            emb_src_trg_weight_sharing=False,  # encoder和decoder的embedding共享参数
            scale_emb_or_prj='prj',
            ):

        super().__init__()   #继承父类的init

        self.model_1 =  SE3Transformer(
            dim = se3_dim,
            heads = se3_heads,
            depth = se3_depth,
            dim_head = se3_dim_head,
            num_degrees =se3_num_degrees,
            valid_radius = se3_valid_radius,
            reduce_dim_out = se3_reduce_dim_out )

        self.model_2 = TFmodels.Transformer(
            n_src_vocab=n_src_vocab,n_trg_vocab=n_trg_vocab, 
            src_pad_idx = src_pad_idx, trg_pad_idx=trg_pad_idx,
            d_word_vec = d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, n_position_src=n_position_src,
            n_position_trg=n_position_trg,batch=batch,trg_pad_len =trg_pad_len)
        
        self.fc1 = nn.Linear( 1024 ,se3_dim)
        self.fc2 = nn.Linear( 1024 ,se3_dim)
        self.fc3 = nn.Embedding(30, d_word_vec, padding_idx=0)
     
    
    def forward(self, features,mask,pep_features,pep_mask,se3_mask,pro_coords):
    
        #features = self.fc1(features)
        #pep_features = self.fc2(pep_features)
        #features = self.fc3(features)
        model1_out1 = self.model_1(features,pro_coords,se3_mask)
        #print(model1_out1.shape)
        #src 蛋白50
        src = model1_out1['0']
        #print(src.shape)
        #src = torch.squeeze(src, dim=3)
        src = torch.reshape(src,((-1,50,20)))
        #print(type(src))
        #print("s",src)

        #print("t",trg)



        #trg 多肽10
        #print(pep_features)

        #print(trg)
        mask_2 = mask.detach().clone().long()
        pep_mask_2=pep_mask.detach().clone().long()
        pep_features =self.fc1(pep_features)
        #0的话就还是0，不是0就不是0作为mask
        # src_pad = src.max(axis=1)+src.argmax(axis=1)#最大列号
        # trg_pad = trg.max(axis=1)+trg.argmax(axis=1)
        prediction = self.model_2(src, pep_features,mask_2,pep_mask_2)



        #prediction = self.model_2(src, src,mask,mask)

        return prediction



