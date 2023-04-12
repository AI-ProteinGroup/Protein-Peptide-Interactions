#import 模块：导入一个模块；注：相当于导入的是一个文件夹，是个相对路径。
#from…import：导入了一个模块中的一个函数；注：相当于导入的是一个文件夹中的文件，是个绝对路径


import numpy as np

import os,sys,argparse,time,random
import torch
import argparse


import torch

from torch.utils.data import DataLoader
import torch.nn as nn
import timeit
from sklearn import metrics
 
import torch.nn as nn
from dataloader import MyDataset  #导入我的dataset
from mynet import MyNet
import tqdm
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# n_src_vocab, #词库大小
# n_trg_vocab, #词库大小,20种，先按23来
# src_pad_idx, #填充元素
# trg_pad_idx, #填充元素
# d_word_vec=512,   # word_vec = d_model 
# d_model=512, 
# d_inner=2048,
# n_layers=6, 
# n_head=8, 
# d_k=64, 
# d_v=64, 
# dropout=0.1, 
# n_position_src=2050,#位置编码个数
# n_position_trg=200,#位置编码个数
# batch=3,
# trg_pad_len=6,#填充长度

#********************************************************************************************************************************#
# decoder使用se3的输出，且没有添加位置编码
#********************************************************************************************************************************#



def argparser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--RESUME',  type=bool, default=False,help='RESUME')
    parser.add_argument('--n_src_vocab',  type=int, default='500',help='n_src_vocab')
    parser.add_argument('--n_trg_vocab',  type=int, default='500',help='n_trg_vocab')
    parser.add_argument('--src_pad_idx',  type=int, default='0',help='src_pad_idx')
    parser.add_argument('--trg_pad_idx',  type=int, default='0',help='trg_pad_idx')
    parser.add_argument('--d_word_vec',  type=int, default='20',help='d_word_vec')
    parser.add_argument('--d_model',  type=int, default='20',help='d_model')
    parser.add_argument('--d_inner',  type=int, default='88',help='d_inner')
    parser.add_argument('--n_layers',  type=int, default='6',help='n_layers')
    parser.add_argument('--n_head',  type=int, default='8',help='n_head')
    parser.add_argument('--d_k',  type=int, default='64',help='d_k')
    parser.add_argument('--d_v',  type=int, default='64',help='d_v')
    parser.add_argument('--dropout',  type=float, default='0.1',help='dropout')
    parser.add_argument('--n_position_src',  type=int, default='50',help='n_position_src')
    parser.add_argument('--n_position_trg',  type=int, default='50',help='n_position_trg')
    parser.add_argument('--batch',  type=int, default='16',help='batch')
    parser.add_argument('--trg_pad_len',  type=int, default='15',help='trg_pad_len')

    parser.add_argument('--se3dim',  type=int, default='20',help='dim')
    parser.add_argument('--se3heads',  type=int, default='4',help='heads')
    parser.add_argument('--se3depth',  type=int, default='1',help='depth')
    parser.add_argument('--se3dim_head',  type=int, default='2',help='dim_head')
    parser.add_argument('--se3num_degrees',  type=int, default='4',help='num_degrees')
    parser.add_argument('--se3valid_radius',  type=int, default='6',help='valid_radius')
    parser.add_argument('--se3reduce_dim_out',  type=bool, default=False,help='reduce_dim_out')

    parser.add_argument('--test_num',  type=int, default='0',help='test_num')
    
    return parser

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, learningrate,epoch):
    """Sets the learning rate to the initial LR decayed by 80% every 10 epochs"""
    lr = learningrate * (0.95 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def calculate_performace(test_num, pred_y,  labels, pred_score):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1        
                
                
    if (tp+fn) == 0:
        q9 = float(tn-fp)/(tn+fp + 1e-06)
    if (tn+fp) == 0:
        q9 = float(tp-fn)/(tp+fn + 1e-06)
    if  (tp+fn) != 0 and (tn+fp) !=0:
        q9 = 1- float(np.sqrt(2))*np.sqrt(float(fn*fn)/((tp+fn)*(tp+fn))+float(fp*fp)/((tn+fp)*(tn+fp)))
        
    Q9 = (float)(1+q9)/2
    accuracy = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp + 1e-06)
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    recall = float(tp)/ (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    ppv = float(tp)/(tp + fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    F1_score = float(2*tp)/(2*tp + fp + fn + 1e-06)
    #MCC= ((tp*tn) - (fn*fp)) / np.sqrt(np.float((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))

    fpr, tpr, thresholds = metrics.roc_curve(labels, pred_score, pos_label=1)
    roc_auc_val = metrics.auc(fpr,tpr)
    return tp,fp,tn,fn,accuracy, precision, sensitivity, recall, specificity, MCC, F1_score, Q9, ppv, npv,roc_auc_val

def result(epoch, run_time, loss, accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, Q9, ppv, npv, tp, fp, tn, fn, file_name):
    # with open(file_name, 'w') as f:
    #     f.write('Epoch \t Time(sec) \t Loss_train \t Accuracy \t Precision \t Recall \t Sensitivity \t Specificity \t MCC \t F1-score \t ROC_AUC \t AUC \t Q9 \t PPV \t NPV \t TP \t FP \t TN \t FN\n')
    run_time = str(run_time)[7:13]
    with open(file_name, 'a') as f:
        result = map(str, [epoch,  run_time, loss, accuracy, precision, recall, sensitivity, specificity, MCC, F1_score,  Q9, ppv, npv, tp, fp, tn, fn,roc_auc_val])
        f.write('\t'.join(result) + '\n')

        #对于每个batch是使用交叉熵计算的，然后optim是使用adam，对于每个epoch我是计算loss的平均值然后输出的
        #保存模型是只保存了state_dict
        #传入数据是使用argparser送到args里，然后MyNet(args).cuda()
#def train(args,device,train_loader,optimizer,epoch_num,learningrate,network,cross_number):
def train(args,device,train_loader,test_loader,optimizer,epoch_num,learningrate,network,cross_number):
        #Before train
        if not args.RESUME:
            with open('./save/'+str(args.test_num)+'/evalue_result.txt', 'w') as f:
                f.write('Epoch \t Time(sec) \t Loss_train \t Accuracy \t Precision \t Recall \t Sensitivity \t Specificity \t MCC \t F1-score \t AUC \t Q9 \t PPV \t NPV \t TP \t FP \t TN \t FN \t ROC_AUC\n')
            with open('./save/'+str(args.test_num)+'/test_result.txt', 'w') as f:
                f.write('Epoch \t Time(sec) \t Loss_train \t Accuracy \t Precision \t Recall \t Sensitivity \t Specificity \t MCC \t F1-score \t AUC \t Q9 \t PPV \t NPV \t TP \t FP \t TN \t FN \t ROC_AUC\n')
        
        writer = SummaryWriter(log_dir='logs/'+str(args.test_num), flush_secs=60)


        model = MyNet(
                args.se3dim,
                args.se3heads,
                args.se3depth,
                args.se3dim_head,
                args.se3num_degrees,
                args.se3valid_radius,
                args.se3reduce_dim_out,
                args.n_src_vocab, #词库大小
                args.n_trg_vocab, #词库大小
                args.src_pad_idx, #填充元素
                args.trg_pad_idx, #填充元素
                args.d_word_vec,   # word_vec = d_model 
                args.d_model, 
                args.d_inner,
                args.n_layers, 
                args.n_head, 
                args.d_k, 
                args.d_v, 
                args.dropout, 
                args.n_position_src,#位置编码个数
                args.n_position_trg,#位置编码个数
                batch=args.batch,
                trg_pad_len =args.trg_pad_len
                ).to(device)

        if optimizer =='SGD':
            optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=learningrate,momentum=0.9)
        elif optimizer == 'Adam':
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=learningrate)
            #optim = torch.optim.Adam(list(model.model_1.parameters())+list(model.model_2.parameters()),lr=learningrate)
        criterion = nn.CrossEntropyLoss().to(device)  #交叉熵损失函数
        start_epoch = 0
        if args.RESUME:
            path_checkpoint = "./models/checkpoint/"+str(args.test_num)+"/ckpt_last.pth"  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点

            model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

            optim.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_epoch = checkpoint['epoch']  # 设置开始的epoch
        print('----------Begin to train model of spec to spec at ' +time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\r')
        max_test_auc = 1
        max_test_loss = 10
        max_valid_auc = 1
        all_loss = []
        all_auc = []

        # stage 1: train model 
        for epoch in range(start_epoch,epoch_num):  
            losses = []#loss是一个epoch的
            print('---------------------------------------------------------------------')  
            #train phase 
            t = tqdm.tqdm(train_loader)
            start = timeit.default_timer()
            for data in t:
                #sys.stdout.write('----------Stage1 Epoch: {}/{} Batch: {}/{}'.format(epoch+1,epoch_num,i+1,len(train_loader))+'\r')
                name, feats ,mask ,label_two, pep_feats ,pep_mask,se3_mask,pro_coords= data

                #name = name.to(device)
                feats = feats.to(device)
                mask = mask.to(device)
                se3_mask = se3_mask.to(device)
                pro_coords = pro_coords.to(device)
                #pep_coors = pep_coors.to(device)
                pep_feats = pep_feats.to(device)
                pep_mask = pep_mask.to(device)

                label = label_two.to(device)
 
                optim.zero_grad()           
                model.train()
                output,_ =model( feats ,mask , pep_feats ,pep_mask,se3_mask,pro_coords)

                #输出是多肽的时候计算loss
                #loss = criterion(output,dec_outputs.view(-1).long())
                #print(output)
                #print(name)
                #print(output)
                loss = criterion(output,label)  
                #print(loss)             
                losses.append(loss.item()* len(label))
                loss.backward()
                optim.step()

            end  = timeit.default_timer()
            run_time = end - start
            epoch_loss = sum(losses) * args.batch / len(train_loader) #len（train_loader）和train_data_size含义相同？
            writer.add_scalar('Train loss '+str(cross_number), epoch_loss, epoch) # (number of trainset) / (batch size)
            all_loss.append(epoch_loss)
            print('/n')
            print('----------Training loss: {}'.format(epoch_loss)+'\r')
            # y_true = np.array(y_true)
            # y_predicted = np.array(y_predicted)
            # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predicted, pos_label=1)
            # valid_auc = metrics.auc(fpr,tpr)
            # print('----------Training auc: {}'.format(valid_auc)+'\r')
            checkpoint = {
                            "net": model.state_dict(),
                            'optimizer':optim.state_dict(),
                            "epoch": epoch
                        }
            if not os.path.isdir("./models/checkpoint/"+str(args.test_num)+"/"):
                os.mkdir("./models/checkpoint"+str(args.test_num)+"/")
            torch.save(checkpoint, './models/checkpoint/'+str(args.test_num)+'/ckpt_last.pth')
            #adjust_learning_rate(optim,learningrate,epoch+1)

            #evaluate phase
            #后面先注释
            model.eval()
            y_true =[]#真实得分
            y_predicted =[]#置信度得分
            y_predlabel = []#预测结果
            t1 = tqdm.tqdm(train_loader)
            for data in t1:
                name, feats ,mask ,label_two, pep_feats ,pep_mask,se3_mask,pro_coords= data

                #name = name.to(device)
                feats = feats.to(device)
                mask = mask.to(device)
                se3_mask = se3_mask.to(device)
                pro_coords = pro_coords.to(device)
                #pep_coors = pep_coors.to(device)
                pep_feats = pep_feats.to(device)
                pep_mask = pep_mask.to(device)

                label = label_two.to(device)

                with torch.no_grad():

                    output,predicted_spec=model( feats ,mask , pep_feats ,pep_mask,se3_mask,pro_coords)
                    predicted_spec = predicted_spec.cpu()
                    predicted_spec = np.array(predicted_spec)
                    for predicte in predicted_spec:
                        #print(predicte[0])
                        y_predicted.append(predicte[1])
                        y_predlabel.append(np.argmax(predicte))
                    for i in label_two:
                        y_true.append(i)


            y_true = np.array(y_true)
            y_predicted = np.array(y_predicted)
            y_predlabel = np.array(y_predlabel)

            tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, F1_score, Q9, ppv, npv,roc_auc_val = calculate_performace(len(y_true) , y_predlabel,  y_true, y_predicted)
            file_result = './save/'+str(args.test_num)+'/evalue_result.txt'
            result(epoch, run_time, loss,  accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, roc_auc_val, Q9, ppv, npv, tp, fp, tn, fn, file_result)
        
            writer.add_scalar('Evaluate auc'+str(cross_number), roc_auc_val, epoch)
            all_auc.append(roc_auc_val)
            print('----------valid auc: {}'.format(roc_auc_val)+'\r')
            if roc_auc_val > max_valid_auc or max_valid_auc ==1:
                pt_name = 'save/'+str(args.test_num)+'/tanh_no.pt'
                torch.save(model.state_dict(),pt_name)  
                #torch.save(model.state_dict(),'model_saved/test.pt')      #保存模型，文件名为训练起始时间
                #print('----------Max auc:{} valid auc:{}, Model was saved in dir /save/'.format(max_valid_auc,valid_auc)+ str(cross_number) +'.pt') 
                print('----------Max auc:{} valid auc:{}, Model was saved in dir /save/{}/tanh_no.pt'.format(max_valid_auc,roc_auc_val,str(args.test_num))) 
                max_valid_auc = roc_auc_val
            

            #test
            model_test = MyNet(
                        args.se3dim,
                        args.se3heads,
                        args.se3depth,
                        args.se3dim_head,
                        args.se3num_degrees,
                        args.se3valid_radius,
                        args.se3reduce_dim_out,
                        args.n_src_vocab, #词库大小
                        args.n_trg_vocab, #词库大小
                        args.src_pad_idx, #填充元素
                        args.trg_pad_idx, #填充元素
                        args.d_word_vec,   # word_vec = d_model 
                        args.d_model, 
                        args.d_inner,
                        args.n_layers, 
                        args.n_head, 
                        args.d_k, 
                        args.d_v, 
                        args.dropout, 
                        args.n_position_src,#位置编码个数
                        args.n_position_trg,#位置编码个数
                        batch=args.batch,
                        trg_pad_len =args.trg_pad_len
                        ).to(device)
           
            model_test.load_state_dict(torch.load('./save/'+str(args.test_num)+'/tanh_no.pt'), strict=True)
            model_test.eval()
            y_true =[]
            y_predicted =[]
            test_losses=[]
            y_predlabel =[]
            t2 = tqdm.tqdm(test_loader)
            #start = timeit.default_timer()
            for data in t2:
                name, feats ,mask ,label_two, pep_feats ,pep_mask,se3_mask,pro_coords= data

                #name = name.to(device)
                feats = feats.to(device)
                mask = mask.to(device)
                se3_mask = se3_mask.to(device)
                pro_coords = pro_coords.to(device)
                #pep_coors = pep_coors.to(device)
                pep_feats = pep_feats.to(device)
                pep_mask = pep_mask.to(device)

                label = label_two.to(device)
                with torch.no_grad():

                    output,predicted_spec=model_test( feats ,mask , pep_feats ,pep_mask,se3_mask,pro_coords)
                    loss = criterion(predicted_spec,label)                  
                    test_losses.append(loss.item()* len(label))
                    predicted_spec = predicted_spec.cpu()
                    predicted_spec = np.array(predicted_spec)
                    for predicte in predicted_spec:
                            #print(predicte[0])
                        y_predicted.append(predicte[1])
                        y_predlabel.append(np.argmax(predicte))
                    for i in label_two:
                        y_true.append(i)

            y_true = np.array(y_true)
            y_predicted = np.array(y_predicted)
            y_predlabel = np.array(y_predlabel)

            tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, F1_score, Q9, ppv, npv,test_auc = calculate_performace(len(y_true), y_predlabel,  y_true, y_predicted)
            file_result = './save/'+str(args.test_num)+'/test_result.txt'
            result(epoch, run_time, loss,  accuracy, precision, recall, sensitivity, specificity, MCC, F1_score, test_auc, Q9, ppv, npv, tp, fp, tn, fn, file_result)
        


            test_loss = sum(test_losses) * args.batch / len(test_loader) 
            writer.add_scalar('Test loss'+str(cross_number), test_loss, epoch)

            writer.add_scalar('Test auc'+str(cross_number), test_auc, epoch)
            print('----------test auc: {}'.format(test_auc)+'\r')
            if test_auc > max_test_auc or max_test_auc ==1:
                max_test_auc = test_auc
                max_test_loss = test_loss





        with open("./save/"+str(args.test_num)+"/auc.txt",'w') as fff:
            string = ''
            for i in all_auc:
                string = string + ' ' + str(i)
            fff.write(string)
        print('**************************************************************************')
        print( 'test auc:{},test loss:{}'.format(max_test_auc,max_test_loss))
        print('**************************************************************************')
        with open("./save/"+str(args.test_num)+"/loss.txt",'w') as ff:
            string = ''
            for i in all_loss:
                string = string + ' ' + str(i)
            ff.write(string)
        print('----------Finish training model of spec to spec at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))



import warnings

if __name__ == "__main__":
# 设置随机数种子
    setup_seed(10)
    warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")
    parser = argparser()
    args = parser.parse_args()
    batch_size= args.batch
    shuffle = True


    print('----------Load Device')
    if torch.cuda.is_available():
        #device = torch.device('cpu')
        device = torch.device('cuda:0')
        print('----------Use Gpu')
    else :
        device = torch.device('cpu')

    train_path = "../new_data/se3_mynet.jsonl"
    split_path = "../new_data/pocketonly_classed_five.npy"

    
    path_all  =[0,1,2,3,4]
    for cross_number in range(1):
        print('----------Cross validation', cross_number)
        #读取数据
        test_path = path_all[args.test_num]
        path_all.remove(test_path)
        test_list = [test_path]
        #新地址 注意地址是反的
        # train_path=['./new_data/1']

        # valid_path = ['./pro_1.csv']
        # test_path = ['./pro_0.csv']
        # train_path= ['./pro_2.csv']
        #train_path = ['./new_data/0']
        print('----------Begin to load dataset')
        train_data = MyDataset(train_path,split_path,path_all)
        #train_dataloader = DataLoader(train_data,batch_size=batch_size,pin_memory=True)
        train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle=shuffle,pin_memory=True)
        #valid_data = MyDataset(path_all)
        #valid_dataloader = DataLoader(valid_data,batch_size=batch_size,pin_memory=True)

        test_data = MyDataset(train_path,split_path,test_list)
        test_dataloader = DataLoader(test_data,batch_size=batch_size,pin_memory=True)
        print('----------Data loaded')



        #测试，查看dataloader效果
        # for i,j,k,l in valid_dataloader:
        #     print(i)
        #     #print(i)
        #     print(j)
        #     print(k)
        #     print(l)
        # xx = 0
        # for x,j,k,l,m,n in  test_dataloader:
        #     print(x)
        #     print(j)
        #     print(k)
        #     print(l)
        #     print(m)
        #     print(n)
        #     break

        #train(args,device,train_dataloader,'Adam',500,0.00005,'MyNet',cross_number)


        train(args,device,train_dataloader,test_dataloader,'Adam',200,0.000005,'MyNet',cross_number)


