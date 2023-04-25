from densenet import DenseNet
from mydataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
import time

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

def validate(model,data_loader):
    print('Validating model...',flush=True)
    model.eval()
    test_meter=AverageMeter()
    with torch.no_grad():
        for data in data_loader:
            img,label=data
            output=model(img)
            loss=F.multilabel_soft_margin_loss(output,label)
            test_meter.add({'loss': loss.item()})
    model.train()
    loss=test_meter.pop('loss')
    print('loss %.4f' % (loss))
    return loss

def train_model():
    model = DenseNet()
    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5055902), (0.23193231))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4722708), (0.22180891))
    ])
    batch_size = 64
    train_dataset = MyDataset("/Users/xuchengtian/code/chest-diseases-classification/dataset/train.txt",transform=transform_train)
    train_data_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,drop_last=True)
    test_dataset = MyDataset("/Users/xuchengtian/code/chest-diseases-classification/dataset/test.txt",transform=transform_test)
    test_data_loader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size,drop_last=True)

    # img, label = next(iter(train_data_loader))
    # print(img.shape)
    # print(label.dtype)
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_meter=AverageMeter()
    for epoch in range(2):
        print("Epoch "+str(epoch+1)+' :')
        for batch_idx,data in enumerate(train_data_loader):
            img,label=data
            optimizer.zero_grad()
            outputs = model(img)
            loss = F.multilabel_soft_margin_loss(outputs,label)
            loss.backward()
            train_meter.add({'loss': loss.item()})
            optimizer.step()

            if (batch_idx+1)%100==0 or True:
                print('batch count: %d loss:%.4f' % (batch_idx+1, train_meter.pop('loss')))
        
        test_loss = validate(model,test_data_loader)
        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state,'/Users/xuchengtian/code/chest-diseases-classification/models/'+
                   time.strftime("%d:%H:%M:%S",time.localtime())+'-epoch'+str(epoch+1)+'.pth')
        print('successfully save model, current loss = %.4f' %(test_loss))



if __name__ == '__main__':
    train_model()
    # print(torch.cuda.is_available())
    # print(time.strftime("%m/%d-%H:%M:%S",time.gmtime()))