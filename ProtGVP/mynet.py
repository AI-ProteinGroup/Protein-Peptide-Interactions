import torch
import torch.nn as nn
import numpy as np
import transformer.Models as TFmodels
#from se3_transformer_pytorch import SE3Transformer
import torch.nn.functional as F
import gvp.data, gvp.models
#SE3Transformer
class ProteinProteinInteractionPrediction(nn.Module):
    def __init__(self,dim=20,n_fingerprint =297180 +100,layer_gnn =6):
        super(ProteinProteinInteractionPrediction, self).__init__()
        self.layer_gnn = layer_gnn
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_gnn             = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        
    def gnn(self, xs1, A1):
        for i in range(self.layer_gnn):
            hs1 = torch.relu(self.W_gnn[i](xs1))            
            xs1 = torch.matmul(A1, hs1)#矩阵乘法
        return xs1
    
    # def mutual_attention(self, h1, h2):
    #     x1 = self.W1_attention(h1)
    #     x2 = self.W2_attention(h2)
    #     print(x1.shape)
    #     print(x2.shape)
    #     m1 = x1.size()[0]
    #     m2 = x2.size()[0]
        
    #     c1 = x1.repeat(1,m2).view(m1, m2, dim)
    #     c2 = x2.repeat(m1,1).view(m1, m2, dim)

    #     d = torch.tanh(c1 + c2)
    #     alpha = torch.matmul(d,self.w).view(m1,m2)
        
    #     b1 = torch.mean(alpha,1)
    #     p1 = torch.softmax(b1,0)
    #     s1 = torch.matmul(torch.t(x1),p1).view(-1,1)
        
    #     b2 = torch.mean(alpha,0)
    #     p2 = torch.softmax(b2,0)
    #     s2 = torch.matmul(torch.t(x2),p2).view(-1,1)
        
    #     return torch.cat((s1,s2),0).view(1,-1), p1, p2
    
    def forward(self, fingerprints1, adjacency1):

        
        """Protein vector with GNN."""
        x_fingerprints1        = self.embed_fingerprint(fingerprints1)#编码层，将所有的子图->dim
        x_protein1 = self.gnn(x_fingerprints1, adjacency1)
        return x_protein1
    
    # def __call__(self, data, train=True):
        
    #     inputs, t_interaction = data[:-1], data[-1]
    #     z_interaction, p1, p2 = self.forward(inputs)
        
    #     if train:
    #         loss = F.cross_entropy(z_interaction, t_interaction)
    #         return loss
    #     else:
    #         z = F.softmax(z_interaction, 1).to('cpu').data[0].numpy()
    #         t = int(t_interaction.to('cpu').data[0].numpy())
    #         return z, t, p1, p2#z预测t实际


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
        
        # self.gnn = ProteinProteinInteractionPrediction(
        #     dim = se3_dim,n_fingerprint = n_fingerprint)#需要修改


        self.fc1 = nn.Linear( 1024 ,100)
        self.fc2 = nn.Linear( 1024 ,100)

     
    
    def forward(self,  h_V,edge_index,h_E,pro_seq,pro_mask, pep_features ,pep_mask):
        
        # print(len(h_V))
        # print(len(h_E))
        features = self.gvp(h_V,edge_index,h_E,pro_seq)
        # print(type(features))
        # print(features.shape)
        # print(type(features))
        features = features.unsqueeze(0)
        # print("before_features",features.shape)
        # print(features)
        features = F.pad(features, pad=(0,0,0,50-len(pro_seq),0,0), mode="constant",value=0)
        # print("features",features.shape)
        # print(features)
        pep_features = pep_features.unsqueeze(0)
        pep_features = self.fc2(pep_features)
        #print("pep_features",pep_features.shape)
        mask_2 = pro_mask.detach().clone().long().unsqueeze(0)
        pep_mask_2=pep_mask.detach().clone().long().unsqueeze(0)
        # print(type(pep_features))
        # print(type(mask_2))
        # print(type(pep_mask_2))
        # print("mask_2",mask_2.shape)
        # print(mask_2)
        # print("pep_mask_2",pep_mask_2.shape)
        # print(pep_mask_2)
        prediction,prediction_softmax = self.model_2(features, pep_features,mask_2,pep_mask_2)



        #prediction = self.model_2(src, src,mask,mask)

        return prediction,prediction_softmax



