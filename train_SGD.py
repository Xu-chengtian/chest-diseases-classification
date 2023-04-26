from densenet import DenseNet
from mydataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
from tqdm import tqdm
import wandb
import os
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss

class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]
    
    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

def validate(model,data_loader,device,batch_size,epoch):
    print('Validating model...',flush=True)
    model.eval()
    test_meter=AverageMeter()
    with torch.no_grad():
        for data in data_loader:
            img,label=data
            img=img.to(device)
            label=label.to(device)
            output=model(img)
            output=output.cpu()
            label=label.cpu()
            ans = torch.where(output>0.5,1,0)

            T = torch.eq(ans,label)
            # TP= torch.sum(torch.mul(T,label),dim=1)
            # P = torch.sum(ans,dim=1)
            # TPFN = torch.sum(label, dim=1)
            test_label_accuracy = torch.sum(torch.sum(T,dim=1))/(batch_size*15)
            # all_accurate = torch.sum(torch.where(torch.sum(T,dim=1)==15,1,0))/batch_size
            # test_precision = torch.sum(TP/P)/batch_size
            # test_recall = torch.sum(TP/TPFN)/batch_size
            # test_F1_score = 2*test_precision*test_recall/(test_precision+test_recall)

            loss=F.multilabel_soft_margin_loss(output,label)
            # test_meter.add({'test_accuracy': test_accuracy})
            # test_meter.add({'all_accurate': all_accurate})
            # test_meter.add({'loss': loss.item()})
            # test_meter.add({'test_precision': test_precision})
            # test_meter.add({'test_recall': test_recall})
            # test_meter.add({'test_F1_score': test_F1_score})

            test_meter.add({'test_accuracy': accuracy_score(label,ans)})
            test_meter.add({'test_label_accuracy': test_label_accuracy})
            test_meter.add({'test_0-1_loss': zero_one_loss(label,ans)})
            test_meter.add({'test_loss': loss.item()})
            test_meter.add({'test_precision': precision_score(label,ans,average='samples')})
            test_meter.add({'test_recall': recall_score(label,ans,average='samples')})
            test_meter.add({'test_F1_score': f1_score(label,ans,average='samples')})
            test_meter.add({'test_hamming_loss': hamming_loss(label,ans)})


    model.train()
    test_accuracy=test_meter.pop('test_accuracy')
    # all_accurate=test_meter.pop('all_accurate')
    test_label_accuracy=test_meter.pop('test_label_accuracy')
    test_loss=test_meter.pop('test_loss')
    test_01_loss=test_meter.pop('test_0-1_loss')
    test_hamming_loss=test_meter.pop('test_hamming_loss')
    test_precision=test_meter.pop('test_precision')
    test_recall=test_meter.pop('test_recall')
    test_F1_score=test_meter.pop('test_F1_score')

    log_test={}
    log_test['epoch']=epoch+1
    log_test['test_loss']=test_loss
    log_test['test_accuracy']=test_accuracy
    log_test['test_label_accuracy']=test_label_accuracy
    # log_test['all_accurate']=all_accurate
    log_test['test_01_loss']=test_01_loss
    log_test['test_precision']=test_precision
    log_test['test_recall']=test_recall
    log_test['test_F1_score']=test_F1_score
    log_test['test_hamming_loss']=test_hamming_loss

    # print('loss %.4f' % (loss))
    return log_test

def train_model(project_name):
    model = DenseNet()
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5055902), (0.23193231))
    ])
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4722708), (0.22180891))
    ])
    batch_size = 64
    train_dataset = MyDataset(os.path.join(os.getcwd(),'dataset','train.txt'),transform=transform_train)
    train_data_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,drop_last=True)
    test_dataset = MyDataset(os.path.join(os.getcwd(),'dataset','test.txt'),transform=transform_test)
    test_data_loader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size,drop_last=True)

    full_train_log = pd.DataFrame()
    full_test_log = pd.DataFrame()
    # img, label = next(iter(train_data_loader))
    # print(img.shape)
    # print(label.dtype)
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_meter=AverageMeter()
    for epoch in range(20):
        print("Epoch "+str(epoch+1)+' :')
        for batch_idx,data in enumerate(train_data_loader):
            img,label=data
            img=img.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = F.multilabel_soft_margin_loss(outputs,label)
            loss.backward()
            train_meter.add({'loss': loss.item()})
            optimizer.step()

            if (batch_idx+1)%50==0:
                print('batch count: %d loss:%.4f' % (batch_idx+1, train_meter.pop('loss')))

            log_train={}
            log_train['epoch']=epoch+1
            log_train['batch']=batch_idx+1
            log_train['train_loss']=loss.item()

            full_train_log = full_train_log.append(log_train, ignore_index=True)
            wandb.log(log_train)

        log_test = validate(model,test_data_loader, device, batch_size, epoch)
        full_test_log = full_test_log.append(log_test, ignore_index=True)
        wandb.log(log_test)

        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state,os.path.join(os.getcwd(),'models',project_name,time.strftime("%d-%H-%M-%S",time.localtime())+'-epoch'+str(epoch+1)+'.pth'))
        torch.cuda.empty_cache()
        print('successfully save model')
    
    full_train_log.to_csv(os.path.join(os.getcwd(),'log',project_name,'train_log.csv'), index=False)
    full_test_log.to_csv(os.path.join(os.getcwd(),'log',project_name,'test_log.csv'), index=False)



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    project_name=time.strftime('%m%d%H%M%S')
    wandb.init(project='chest-diseases-classification', name=project_name)
    os.makedirs(os.path.join(os.getcwd(),'models',project_name))
    os.makedirs(os.path.join(os.getcwd(),'log',project_name))
    train_model(project_name)

    # print(torch.cuda.is_available())
    # print(time.strftime("%m/%d-%H:%M:%S",time.gmtime()))

    # model = DenseNet()
    # transform_train = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5055902), (0.23193231))
    # ])
    # batch_size = 8
    # train_dataset = MyDataset(os.path.join(os.getcwd(),'dataset','train.txt'),transform=transform_train)
    # train_data_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,drop_last=True)

    # img, label = next(iter(train_data_loader))
    # print(img.shape)

    # output = model(img)
    # ans = torch.where(output>0.5,1,0)
    # print(ans)
    # print(label)

    # print(accuracy_score(label,ans))
    # print(zero_one_loss(label,ans))
    # print(precision_score(label,ans,average='samples'))
    # print(recall_score(label,ans,average='samples'))
    # print(hamming_loss(label,ans))

    # T = torch.eq(ans,label)
    # TP= torch.sum(torch.mul(T,label),dim=1)
    # P = torch.sum(ans,dim=1)
    # TPFN = torch.sum(label, dim=1)
    # print(TP/P)
    # print(TP/TPFN)
    
    # test_accuracy = torch.sum(torch.sum(T,dim=1))/(batch_size*15)
    # all_accurate = torch.sum(torch.where(torch.sum(T,dim=1)==15,1,0))/batch_size
    # test_precision = torch.sum(TP/P)/batch_size
    # test_recall = torch.sum(TP/TPFN)/batch_size
    # test_F1_score = 2*test_precision*test_recall/(test_precision+test_recall)
    # print(test_accuracy,all_accurate,test_precision,test_recall,test_F1_score)
    
    # transform_test = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4722708), (0.22180891))
    # ])
    # batch_size = 64
    # test_dataset = MyDataset(os.path.join(os.getcwd(),'dataset','test_128.txt'),transform=transform_test)
    # test_data_loader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size,drop_last=True)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # test=validate(DenseNet().to(device),test_data_loader,device,64,1)
    # print(test)
    