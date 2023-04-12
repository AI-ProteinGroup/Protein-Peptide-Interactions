
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import os
import torch
import re
import random
import jsonlines
import json
#import difflib
#from sklearn.preprocessing import OneHotEncoder 
#from tape import ProteinBertModel, ProteinBertConfig, TAPETokenizer  # type: ignore
class MyDataset(Dataset):

    def __init__(self,path,split_path,which_data):
        

        self.max_pep_length = 10#多肽
        self.max_seq_length = 50#蛋白


    #对于每个口袋先计算坐标，编码，填充，mask，然后用字典保存信息，mask为最大长度(事先准备好),然后用大字典保存index
    #先当成会自动添加batch维
        def _load_data(self,path,split_path,which_data):
            dataset = {}
            index = 0
            with open(split_path) as f:
                dataset_splits = json.load(f)
                need_list = dataset_splits[which_data]
            with open(path, "r+", encoding="utf8") as ff:
                for data in jsonlines.Reader(ff): 
                    if data['name'] in need_list:
                        dataset[index] = data
                        index = index +1
            return dataset

        self.dataset = _load_data(self,path,split_path,which_data) #加载模型





    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        # todo 根据索引idx取数据,处理数据并输出成模型需要的形状
        #if index in self.cache: #如果有，直接从cache里取出
        #到label去了，有问题加
        pro_features =self.dataset[idx]['pro_feature']
        
        pro_seq = self.dataset[idx]['seq']

        pro_mask = self.dataset[idx]['pro_mask']

        pep_feats = self.dataset[idx]['pep_feature']

        pep_mask = self.dataset[idx]['pep_mask'] 
        label =self.dataset[idx]['label']
        name = self.dataset[idx]['name']

        pro_features = np.array(pro_features,dtype='float32')
        pro_features = np.pad(pro_features,((0,self.max_seq_length-len(pro_features)),(0,0)),'constant',constant_values=(0,0)).astype(np.float32)
        pro_mask = np.array(pro_mask)
        pep_feats =  np.array(pep_feats,dtype='float32')
        pep_mask = np.array(pep_mask)

        return pro_features ,pro_seq,pro_mask, pep_feats,pep_mask,label




if __name__ == "__main__":

    batch_size= 1
    shuffle = True

 
    #read data
    train_path  = ['./new_data/0']
    print('----------Begin to load dataset')
    train_data = MyDataset(train_path)
    train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle=shuffle,pin_memory=True)
    xx = 0
    for i,j,k,l,m,n in  train_dataloader:

        print(i.shape)
        print(i)
        print(j.shape)
        print(j)
        print(k.shape)
        print(k)
        print(l.shape)
        print(l)
        print(m.shape)
        print(m)
        print(n.shape)
        print(n)        
        xx = xx + 1
        if xx >= 10 :
            break
        #break
      
        # print("feats.shape:",j)
        # print("coors.shape:",i)
        # print("mask.shape:",k)
        # peptide_code = torch.tensor([tokenizer.encode(peptide_code)])
        # print(peptide_code)
        # peptide_code = modelmin(m,n)[1]
        # print(peptide_code)
        # print(peptide_code.shape)
        # peptide_code = modelmin(m)[0]
        # print(peptide_code)




