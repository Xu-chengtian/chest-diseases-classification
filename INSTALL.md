# Installation

## 源码下载
```
git clone https://github.com/Xu-chengtian/chest-diseases-classification.git
```

## 主要依赖库

- Python 3.10
- PyTorch 
- Torchvision
- Numpy
- Pillow
- Pandas
- wandb
- torchinfo

<b>

### 注意:

- wandb 仅可使用pip工具下载安装
- 如果想要使用 GPU 训练，请下载支持 GPU 的Pytorch <i>(推荐)</i>
- 使用 anaconda 实现虚拟环境生成以及包体管理 <i>(推荐)</i>
</b>

## 数据集
`chest diseases classification` 使用 [NIH Chest X-Ray](https://www.kaggle.com/datasets/nih-chest-xrays/data) 作为数据集。

也可以借助 [百度飞桨平台](https://aistudio.baidu.com/aistudio/datasetdetail/35660) 快速下载

<b><i>百度飞桨平台可能存在数据缺失问题</b></i>

下载后文件存放在dataset文件夹下，并按照 <b>dataset/image1/images/*.png</b> 结构存放