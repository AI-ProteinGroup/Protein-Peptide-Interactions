''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformer.Layers import EncoderLayer, DecoderLayer

def get_need_data(mask,d_word_vec,batch,dec_output):
    index=torch.arange(0,d_word_vec).repeat([len(dec_output),1])
    ##print(index)
    batch_index=torch.arange(0,len(dec_output))
    
    mask = mask.squeeze()
    if len(dec_output)==1:
        mask = mask.unsqueeze(0)
        # print(dec_output.shape)
        # print(mask.shape)
        # print(index.shape)
    index[~mask]=-1
    ##print(index)
    final_index = index.argmax(axis=1)
    #print(final_index)
    ##print(dec_output[batch_index,final_index,:])
    return dec_output[batch_index,final_index,:]


def get_pad_mask(seq, pad_idx=0):
    # 一个batch多个句子长度不同，为了一次训练多个句子，得pad到相同长度。 mask用于标记出哪几个字符是不是pad字符。
    # seq     batch_size x max_len
    # output  batch_size x max_len 其中pad的位置为false， 不是pad的为true

    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    # 一个batch的shape是 batch_size x max_seq_len
    # 那么，每个句子都会被用 max_seq_len x max_seq_len的对角阵做mask
    # 比如，最长的句子长度为4
    # subsequent_mask =  1 0 0 0
    #                    1 1 0 0
    #                    1 1 1 0
    #                    1 1 1 1

    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    def __init__(self, 
                d_word_vec,   # word2vec的维度
                n_position=200): # 一般为最长句子长度
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_word_vec))

    def _get_sinusoid_encoding_table(self, n_position, d_word_vec):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_word_vec) for hid_j in range(d_word_vec)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        # shape_of_postable  1 x n_position x d_word_vec
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(
            self, 
            n_src_vocab, # Encode语言词库大小
            d_word_vec,  # word_vec维度
            n_layers, 
            n_head, 
            d_k, 
            d_v,
            d_model,        # transformer各层的输入和输出的词向量维度
            d_inner,        # transformer内部全连接层维度
            pad_idx, 
            dropout=0.1, 
            n_position=200, # 位置编码个数。 一般设置为大于最长句子的长度
            scale_emb=False):

        super().__init__()

        #self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx) 
        #self.src_word_emb = EmptyDo()
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        
        enc_slf_attn_list = []

        # -- Forward
        enc_output = src_seq        # word2vec   batch_size x maxlen x d_embedding 
        # print("enc_output-emb")
        # print(enc_output.shape)
        ##print(enc_output)
        ##print(enc_output)
        if self.scale_emb:                               # embbeding scale
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(enc_output)  # positioned encode + dropout
        # print("enc_output-position")    
        # print(enc_output.shape) 
        ##print(enc_output)   
        enc_output = self.layer_norm(enc_output)                  # layer norm  -> final embedding
        # print("enc_output-ln")    
        # print(enc_output.shape) 
        ##print(enc_output) 

        # enc_output 作为输入 其维度为 batch_size x maxlen x d_embedding 
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)    # input -> input 
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []                 # receive attention matrix
        # print("enc_output-fin")    
        # print(enc_output.shape) 
        ##print(enc_output) 

        if return_attns:
            # enc_output     batch x len x d_word2vec
            # enc_slf_attn   layer x batch x n x len x len
            return enc_output, enc_slf_attn_list
        
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, 
            n_trg_vocab, 
            d_word_vec, 
            n_layers, 
            n_head, 
            d_k, 
            d_v,
            d_model, 
            d_inner, 
            pad_idx, 
            n_position=200, 
            dropout=0.1, 
            scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx) # word embedding
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)      # position encode
        self.dropout = nn.Dropout(p=dropout)            
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        # 最终dec_out   b x l x d_model
        #dec_output = self.trg_word_emb(trg_seq) 
        dec_output = trg_seq
        
        # print("dec_output-emb")    
        # print(dec_output.shape) 
        ##print(dec_output) 

        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        
        # dec_output = self.dropout(dec_output) # 不用位置编码
        # print("dec_output-position")    
        # print(dec_output.shape) 
        #print(dec_output) 
        dec_output = self.layer_norm(dec_output)                        # 与encoder一模一样

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            # dec_output             b x seq x l
            # dec_slf_attn_list      layer x b x n x l x l
            # dec_enc_attn_list      layer x b x n x l1 x l2
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        # print("dec_output-fin")    
        # print(dec_output.shape) 
        ##print(dec_output) 
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, 
            n_src_vocab, 
            n_trg_vocab, 
            src_pad_idx, 
            trg_pad_idx,
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
            scale_emb_or_prj='prj'

            ):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.trg_pad_len = trg_pad_len
        self.batch = batch
        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position_src,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position_trg,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        #self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        self.trg_word_prj = nn.Linear(trg_pad_len*d_model, 2 , bias=False)
        self.sigmoid    =  nn.Sigmoid()
        #self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def loss_function(self, dec_output,dec_in,trg_mask_1):
        # print(dec_output)
        # print(dec_in)
        loss_ = self.criterion(dec_output,dec_in)
        # print(loss_.shape)
        
        # print(trg_mask_1.shape)
        loss_*= trg_mask_1.squeeze()
        return torch.mean(loss_)

    def forward(self, src_seq, trg_seq,src_pad,trg_pad):
        # 只能用于seq2seq的训练
        # 输入的 size 为 batch_size X seq_len 的 torch.tensor。这里每个字符用int索引表示，而不是one-hot向量
        # 输出的 size 为 batch_size x seq_len 的 torch.tensor。
        # print(src_seq)
        # print(src_seq.shape)
        # print(trg_seq)
        # print(trg_seq.shape)
        src_mask = get_pad_mask(src_pad, self.src_pad_idx)     # batchsize x 1 x len   是pad的位置为FALSE，不是pad的为True
        # print("src_mask")
        # print(src_mask.shape)
        # print(src_mask)
        #print(src_mask)
        #print(trg_seq.data)
        trg_mask_1 = get_pad_mask(trg_pad, self.trg_pad_idx)  
        # print("trg_mask_1")
        # print(trg_mask_1.shape)
        trg_mask =trg_mask_1 & get_subsequent_mask(trg_pad) # batchsize x len x len
        # print("trg_mask")
        # print(trg_mask.shape)
        # print(trg_mask)
        # enc_output   batch x len x d_word2vec
        ##############################enc_output, *_ = self.encoder(src_seq, src_mask)
        # print(src_seq.shape)
        # print(enc_output.shape)
        dec_output, *_ = self.decoder(trg_seq, trg_mask_1, src_seq, src_mask)
        dec_output=dec_output.view(dec_output.size(0),-1)
        #print(dec_output.shape)
        dec_logits = self.trg_word_prj(dec_output)
        #print(dec_logits.shape)
        # print(trg_mask_1)
        # print(trg_mask_1.shape)
        # print(dec_output.shape)
        # print(len(dec_output))
        #need_data = get_need_data(trg_mask_1,self.trg_pad_len,self.batch,dec_output)
        #seq_logit = self.trg_word_prj(need_data)   # batch_size x len x n_trg_vocab
        #print("seq_logit")    
        #print(seq_logit.shape) 
        #print(dec_logits)
        softmax_prediction = F.softmax(dec_logits,1)#dim =1

        #return seq_logit.view(-1, seq_logit.size(2)) # (batch_size x len) x n_trg_vocab
        return dec_logits,softmax_prediction
