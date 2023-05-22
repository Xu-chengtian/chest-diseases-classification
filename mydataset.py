from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# 自定义数据集类
# 读取自定义的训练集测试集图片和路径

class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        # 初始化，从指定txt文件中逐个读取文件名和标签，存在列表中
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            disease = words[1].split(',')
            # 标签存储为numpy数组
            disease_arr=np.zeros(14, np.float32)
            for i in disease:
                if i == '0':
                    continue
                disease_arr[int(i)-1]=1.0
            imgs.append((words[0], disease_arr))
            self.imgs = imgs 
            # 图像预处理方法存储
            self.transform = transform
            self.target_transform = target_transform
    def __getitem__(self, index):
        # 传入下标，读取图像及其标签
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('L') 
        # 读取图像后进行预处理
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    def __len__(self):
        # 返回数据集中图像数量
        return len(self.imgs)