from PIL import Image
import torch
from mydataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# 打印图像信息
def pic_inf():
    img = Image.open(os.path.join(os.getcwd(),'dataset','images_001','images','00000001_000.png')).convert('L')
    print(img.getbands())
    print(img.split())
    max=0
    min=255
    for i in range(1024):
        for j in range(1024):
            if img.getpixel((i,j))>max:
                max=img.getpixel((i,j))
                tmp_max=[i,j]
            elif img.getpixel((i,j))<min:
                min=img.getpixel((i,j))
                tmp_min=[i,j]
    print(max,min)
    print(tmp_max)
    print(tmp_min)

# 计算训练集和测试集图像的方差和标准差，用于后续预处理进行归一化运算
def cal_mean_variance(data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('dataset count: '+str(len(data)))
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    # 初始化标准差和方差
    mean = torch.zeros(1)
    std = torch.zeros(1)
    # 遍历图片更新参数
    for X, _ in train_loader:
        mean[0] += X[:, 0, :, :].mean()
        std[0] += X[:, 0, :, :].std()
    mean.div_(len(data))
    std.div_(len(data))
    # 返回结果
    print(list(mean.numpy()), list(std.numpy()))

if __name__ == '__main__':
    pic_inf()
    transform = transforms.Compose([
    transforms.ToTensor(),
])
    # 分别对训练集和测试集进行计算
    train_dataset = MyDataset(os.path.join(os.getcwd(),'dataset','train.txt'),transform=transform)
    test_dataset = MyDataset(os.path.join(os.getcwd(),'dataset','test.txt'),transform=transform)
    print('Compute mean and variance for training data.')
    cal_mean_variance(train_dataset)
    print('Compute mean and variance for test data.')
    cal_mean_variance(test_dataset)