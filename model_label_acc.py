from densenet import DenseNet
import torch
import os
import argparse
from mydataset import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# 测试标签准确率
# 后期测试可以使用，在训练过程中已经加入

# AverageMeter类
class AverageMeter():
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

if __name__ == '__main__':
    # 命令行传参设置batch大小，测试图片txt文件路径，模型路径
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b','--batch', type=int,
                        default=32, help='set the test dataset batch size')
    parser.add_argument("--test_path", type=str,
                        default='test.txt', help="setting test txt path")
    parser.add_argument('--model_path',type=str,
                        default='/Users/xuchengtian/code/chest-diseases-classification/models/model.pth',help="setting the model path which you want to test")
    # 解析命令行参数
    args = parser.parse_args()

    if args.model_path == None:
        raise Exception('No model selected! Please try again.')
    # 图像与处理
    net = DenseNet()
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4722708), (0.22180891))
    ])
    test_dataset = MyDataset(os.path.join(os.getcwd(),'dataset',args.test_path),transform=transform_test)
    test_data_loader = DataLoader(test_dataset,shuffle=True,batch_size=args.batch,drop_last=True)
    # 读取模型
    pretrain_model = torch.load(args.model_path)
    net.load_state_dict(pretrain_model['net'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    test_meter=AverageMeter()
    # 遍历测试集计算
    for data in test_data_loader:
        img,label=data
        img=img.to(device)
        label=label.to(device)
        output=net(img)
        output=output.cpu()
        label=label.cpu()
        ans = torch.where(output>0.5,1,0)
        # 预测正确的标签
        T = torch.eq(label,ans)
        disease = (torch.sum(T,dim=0)/args.batch).tolist()
        for idx in range(14):
            test_meter.add({'disease'+str(idx+1):disease[idx]})
        # 将阈值定为0，进行评估
        ans = torch.where(output>0,1,0)
        T = torch.ge(ans,label)
        disease = torch.sum(torch.where(torch.sum(T,dim=1)==14,1,0))/args.batch
        test_meter.add({'attention_disease':disease})
    disease=[]
    for idx in range(14):
        disease.append(test_meter.pop('disease'+str(idx+1)))
    attention_disease=test_meter.pop('attention_disease')
    # 返回两种阈值的准确率
    print(disease)
    print(attention_disease)