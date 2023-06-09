from densenet import DenseNet
from mydataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import wandb
import os
import pandas as pd
import warnings
import argparse
import logging
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss

# 训练程序

# 疾病与标签对应表
diseases={1: 'Atelectasis', 2: 'Cardiomegaly', 3: 'Effusion', 4: 'Infiltration', 5: 'Mass', 
         6: 'Nodule', 7: 'Pneumonia', 8: 'Pneumothorax', 9: 'Consolidation', 10: 'Edema', 
         11: 'Emphysema', 12: 'Fibrosis', 13: 'Pleural_Thickening', 14: 'Hernia', 0: 'No Finding'}

# AverageMeter类，存储并计算各类指标
class AverageMeter():
    # 初始化一个指标计算
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]
    # 为指定指标添加内容
    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1
    # 读取指定指标平均值
    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)
    # 返回指定指标平均值并删除该指标
    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

# 评估函数，在每一轮训练完成后调用，评估当前模型各类指标
def validate(model,data_loader,device,batch_size,epoch,logger):
    logger.info('***Validating model***')
    # 转换为评估模式
    model.eval()
    test_meter=AverageMeter()
    with torch.no_grad():
        # 遍历测试集，计算指标
        for data in data_loader:
            img,label=data
            img=img.to(device)
            label=label.to(device)
            output=model(img)
            output=output.cpu()
            label=label.cpu()
            # 阈值设定为0.5
            ans = torch.where(output>0.5,1,0)
            # 得到预测值和标签相同的所有结果
            T = torch.eq(ans,label)
            # TP= torch.sum(torch.mul(T,label),dim=1)
            # P = torch.sum(ans,dim=1)
            # TPFN = torch.sum(label, dim=1)
            test_label_accuracy = torch.sum(torch.sum(T,dim=1))/(batch_size*15)
            # all_accurate = torch.sum(torch.where(torch.sum(T,dim=1)==15,1,0))/batch_size
            # test_precision = torch.sum(TP/P)/batch_size
            # test_recall = torch.sum(TP/TPFN)/batch_size
            # test_F1_score = 2*test_precision*test_recall/(test_precision+test_recall)
            # 对于每一种疾病统计准确率
            disease = (torch.sum(T,dim=0)/batch_size).tolist()
            for idx in range(14):
                test_meter.add({diseases[idx+1]:disease[idx]})
            # 设置阈值为0再次统计准确率
            att_ans = torch.where(output>0,1,0)
            att = torch.ge(att_ans,label)
            disease = torch.sum(torch.where(torch.sum(att,dim=1)==14,1,0))/batch_size
            test_meter.add({'attention_disease':disease})
            # 计算测试集损失
            loss=F.multilabel_soft_margin_loss(output,label)
            # test_meter.add({'test_accuracy': test_accuracy})
            # test_meter.add({'all_accurate': all_accurate})
            # test_meter.add({'loss': loss.item()})
            # test_meter.add({'test_precision': test_precision})
            # test_meter.add({'test_recall': test_recall})
            # test_meter.add({'test_F1_score': test_F1_score})
            # 将计算结果加入AverageMeter，同时加入其他指标
            test_meter.add({'test_accuracy': accuracy_score(label,ans)})
            test_meter.add({'test_label_accuracy': test_label_accuracy})
            test_meter.add({'test_0-1_loss': zero_one_loss(label,ans)})
            test_meter.add({'test_loss': loss.item()})
            test_meter.add({'test_precision': precision_score(label,ans,average='samples')})
            test_meter.add({'test_recall': recall_score(label,ans,average='samples')})
            test_meter.add({'test_F1_score': f1_score(label,ans,average='samples')})
            test_meter.add({'test_hamming_loss': hamming_loss(label,ans)})

    model.train()
    # 从AverageMeter中得到均值，存放在日志中并同步更新到wandb
    log_test={}
    log_test['epoch']=epoch+1
    log_test['test_loss']=test_meter.pop('test_loss')
    log_test['test_accuracy']=test_meter.pop('test_accuracy')
    log_test['test_label_accuracy']=test_meter.pop('test_label_accuracy')
    # log_test['all_accurate']=test_meter.pop('all_accurate')
    log_test['test_01_loss']=test_meter.pop('test_0-1_loss')
    log_test['test_precision']=test_meter.pop('test_precision')
    log_test['test_recall']=test_meter.pop('test_recall')
    log_test['test_F1_score']=test_meter.pop('test_F1_score')
    log_test['test_hamming_loss']=test_meter.pop('test_hamming_loss')
    for idx in range(14):
        log_test[diseases[idx+1]]=test_meter.pop(diseases[idx+1])
    log_test['attention_disease']=test_meter.pop('attention_disease')

    # print('loss %.4f' % (loss))
    return log_test

# 训练模型主函数
def train_model(pre_time,train_path,test_path,batch_size,epoch_num,change_opt,out_fre,logger,model_path):
    """
        :param pre_time: (str) 当前训练时间
        :param train_path: (str) 训练集文件路径
        :param test_path: (str) 测试集文件路径
        :param batch_size: (int) 批样本数量
        :param epoch_num: (int) 迭代次数
        :param change_opt: (int) 优化器变更位置
        :param out_fre: (int) 打印输出的频率
        :param logger: (logger) 日志流
        :param model_path: (str) 继续训练模型路径
        """
    project_name = pre_time+' batch '+str(batch_size)+' epoch '+str(epoch_num)+' SGD '+str(change_opt)
    # wandb初始化
    wandb.init(project='chest-diseases-classification', name=project_name)
    wandb.config = {"time": pre_time, "batch_size": batch_size, "epochs": epoch_num,
                    "change SGD epoch": change_opt, "log frequency": out_fre}
    wandb.config.update()
    # 建立日志、模型存储文件夹
    os.makedirs(os.path.join(os.getcwd(),'models',project_name))
    os.makedirs(os.path.join(os.getcwd(),'log',project_name))

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
    # 数据集初始化
    train_dataset = MyDataset(os.path.join(os.getcwd(),'dataset',train_path),transform=transform_train)
    train_data_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,drop_last=True)
    test_dataset = MyDataset(os.path.join(os.getcwd(),'dataset',test_path),transform=transform_test)
    test_data_loader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size,drop_last=True)

    full_train_log = pd.DataFrame()
    full_test_log = pd.DataFrame()
    # img, label = next(iter(train_data_loader))
    # print(img.shape)
    # print(label.dtype)
    # 将模型转换为训练模式
    model.train()
    # 判断当前设备是否支持cuda
    if torch.cuda.is_available():
        logger.info('cuda is available, using cuda')
    else:
        logger.info('cuda is not available, using cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_epoch=0
    # 优化器设置和确定学习率更新策略
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)
    # 是否继续训练
    if model_path!=None:
        pretrained = torch.load(model_path)
        model.load_state_dict(pretrained['net'])
        optimizer.load_state_dict(pretrained['optimizer'])
        start_epoch = pretrained['epoch'] + 1

    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    logger.info('using init optimizer: Adam')
    train_meter=AverageMeter()
    # 迭代开始
    for epoch in range(start_epoch,epoch_num):
        # 判断是否需要更换优化器
        if epoch == change_opt-1:
            optimizer = optim.SGD(model.parameters(), lr=optimizer.state_dict()['param_groups'][0]['lr'], momentum=0.5)
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=3, gamma=0.5)
            logger.info('change optimizer from Adam to SGD at epoch '+ str(epoch+1))
        logger.info("Epoch "+str(epoch+1)+' :')
        # 从训练集中不断读取数据，计算并更新参数
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

            if (batch_idx+1)%out_fre==0:
                logger.info('batch count: %d loss:%.4f' % (batch_idx+1, train_meter.pop('loss')))
            # 记录训练过程中参数
            log_train={}
            log_train['learning_rate']=optimizer.state_dict()['param_groups'][0]['lr']
            log_train['epoch']=epoch+1
            log_train['batch']=batch_idx+1
            log_train['train_loss']=loss.item()

            full_train_log = full_train_log.append(log_train, ignore_index=True)
            # 同步到wandb
            wandb.log(log_train)
        
        scheduler.step()

        log_test = validate(model,test_data_loader, device, batch_size, epoch, logger)
        full_test_log = full_test_log.append(log_test, ignore_index=True)
        # 同步测试集指标到wandb
        wandb.log(log_test)
        # 保存模型
        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state,os.path.join(os.getcwd(),'models',project_name,time.strftime("%d-%H-%M-%S",time.localtime())+'-epoch'+str(epoch+1)+'.pth'))
        torch.cuda.empty_cache()

        logger.info('successfully save model')
    # 日志保存
    full_train_log.to_csv(os.path.join(os.getcwd(),'log',project_name,'train_log.csv'), index=False)
    full_test_log.to_csv(os.path.join(os.getcwd(),'log',project_name,'test_log.csv'), index=False)
    wandb.finish()



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # 忽略warning，设置日志格式
    if not os.path.exists(os.path.join(os.getcwd(),'log','log_output')):
        os.makedirs(os.path.join(os.getcwd(),'log','log_output'))
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(os.getcwd(),'log','log_output','log'+time.strftime('%m%d%H%M%S')+'.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.info('Start')
    # 从命令行读取参数
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch', type=int,
                        default=64, help='setting batch size')
    parser.add_argument('-e', '--epoch', type=int,
                        default=20, help='setting train epoch')
    parser.add_argument('-c','--change_opt',type=int,
                        default=0,help='set the epoch num where optimizer will change into SGD')
    parser.add_argument('-s','--step',type=int,
                        default=2,help='set the step of optimizer changing')
    parser.add_argument('-r','--repeat',type=int,
                        default=1,help='train the model several time with different/same method')
    parser.add_argument('-l','--logger',type=int,
                        default=50,help='setting the logger frequence in training output')
    parser.add_argument("--train_path", type=str,
                        default='train.txt', help="setting train txt path")
    parser.add_argument("--test_path", type=str,
                        default='test.txt', help="setting test txt path")
    parser.add_argument('--model_path',type=str,
                        default=None,help="setting the model path which you want to continue training")
    # 解析命令行参数并打印
    args = parser.parse_args()
    logger.info(args)
    # 如果参数设置不合理将提示
    if args.change_opt+(args.repeat-1)*args.step > args.epoch:
        logger.info('change opt num will out of range')
    if args.model_path!=None:
        logger.info('using pretrained model and continue training')
    # 对于多次训练，使用循环实现
    if args.repeat>1:
        for i in range(args.repeat):
            pre_time=time.strftime('%m%d%H%M%S')
            change_opt=args.change_opt+i*args.step
            train_model(pre_time,args.train_path,args.test_path,args.batch,args.epoch,change_opt,args.logger,logger,args.model_path)
    else:
        pre_time=time.strftime('%m%d%H%M%S')
        train_model(pre_time,args.train_path,args.test_path,args.batch,args.epoch,args.change_opt,args.logger,logger,args.model_path)
    logger.info('Finish')

    # 以下为测试用代码

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