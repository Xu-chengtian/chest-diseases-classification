# Chest Disease classification

本仓库是 <b>针对胸部X光片的肺部疾病分类系统设计</b> 模型训练部分源码。

## 下载
请参考 [INSTALL.md](./INSTALL.md) 下载依赖库和数据集。

## 数据集
`chest diseases classification` 使用 [NIH Chest X-Ray](https://www.kaggle.com/datasets/nih-chest-xrays/data) 作为数据集。

### 数据集介绍
Chest X-ray数据集包含30,805名患者的112,120张正面视图的X射线图像，以及利用NLP从相关放射学报告挖掘的14类疾病的图像标签（每个图像可以有多个标签）。数据集含有14类常见的胸部病理，包括肺不张、变实、浸润、气胸、水肿、肺气肿、纤维变性、积液、肺炎、胸膜增厚、心脏肥大、结节、肿块和疝气。


## 训练流程
<u> 请参考论文了解更加详细的训练流程。 </u>

安装完数据集和依赖库后，如果使用conda构建虚拟环境，来到本项目目录，激活虚拟环境。

```
conda activate python310
```
如果希望使用GPU进行训练可以依次输入以下代码测试cuda是否可用。
```
python
import torch
torch.cuda.is_available()
```
如果显示True则表示cuda可用，显示False则请确认torch是否安装gpu版本或下载的torch是否与电脑的GPU型号匹配。

确认dataset下数据集存放结构，以及util文件夹是否存放在仓库目录下。

运行[prepare_train_data.py](./data_tools/prepare_train_data.py)进行数据预处理。
```
python data_tools/prepare_train_data.py
```

如果在dataset文件夹下方出现train.txt和test.txt，且文件内容为录像的绝对路径和标签则表示运行成功。

随后确认训练所需设置的超参数。

可以通过以下命令查看超参数所对应的标签和默认值。
```
python train.py --help
```

如果使用批样本数量为32，迭代次数25次，在第6,8,10次迭代切换优化器，每200个epoch打印一次损失，则参数设置如下。
```
python train.py -b 32 -e 25 -c 6 -s 2 -r 3 -l 200
```

第一次使用需要登录wandb，按照终端输出指令操作即可。

登录完成后通过网页链接即可远程查看训练过程。

<b><i>
注意：请保持网络良好，否则会因为无法同步数据导致训练进度卡死。如网络条件不好请注释掉所有wandb代码或使用离线wandb。
</b></i>

训练完成后，模型存储在model文件夹下，日志保存在log文件夹下，wandb数据保存在wandb文件夹下。

## 系统展示
访问[系统展示页面](www.qiqi77.site:8000)查看展示结果

如访问失败则域名到期或关闭服务，可以通过[系统展示程序](https://github.com/Xu-chengtian/chest-diseases-classification-web)下载代码自行搭建展示页面。

## 注意
本代码由Xu-Chengtian编写。

如需使用请联系<a>xuchengtian0702@163.com</a>。
