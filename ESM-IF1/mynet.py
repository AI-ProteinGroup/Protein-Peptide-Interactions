import torch
import torch.nn as nn
import numpy as np
import transformer.Models as TFmodels
#from se3_transformer_pytorch import SE3Transformer
import torch.nn.functional as F
import gvp.data, gvp.models



class MyNet(nn.Module):
    
    def __init__(
            self, 
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



            gvp_node_dim = (100,16),
            gvp_edge_dim  = (32,1),
            gvp_node_in_dim = (6,3),
            gvp_edge_in_dim = (32,1),

            batch=4,
            trg_pad_len=15
         ):

        super().__init__()   #继承父类的init

        self.gvp =  gvp.models.MQAModel(gvp_node_in_dim,
                                        gvp_node_dim, 
                                        gvp_edge_in_dim, 
                                        (gvp_edge_dim),seq_in = True)

        self.model_2 = TFmodels.Transformer(
            n_src_vocab=n_src_vocab,n_trg_vocab=n_trg_vocab, 
            src_pad_idx = src_pad_idx, trg_pad_idx=trg_pad_idx,
            d_word_vec = d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, n_position_src=n_position_src,
            n_position_trg=n_position_trg,batch=batch,trg_pad_len =trg_pad_len)
        

        self.fc1 = nn.Linear( 512 ,512)
        self.fc2 = nn.Linear( 1024 ,512)

     
    
    def forward(self,  pro_features ,pro_mask, pep_features ,pep_mask):

        pep_features = self.fc2(pep_features)

        prediction,prediction_softmax = self.model_2(pro_features, pep_features,pro_mask,pep_mask)



        return prediction,prediction_softmax



