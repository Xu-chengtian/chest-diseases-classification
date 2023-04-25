from PIL import Image
from torch.utils.data import Dataset
import numpy as np
class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            disease = words[1].split(',')
            disease_arr=np.zeros(15, np.float32)
            for i in disease:
                disease_arr[int(i)]=1.0
            imgs.append((words[0], disease_arr))
            self.imgs = imgs 
            self.transform = transform
            self.target_transform = target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('L') 
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    def __len__(self):
        return len(self.imgs)