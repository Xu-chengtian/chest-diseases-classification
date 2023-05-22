import csv
import os

# 数据集分类
# 根据疾病将所有图像进行分类，存储于dataset/diseases目录下
# 方便挑选图片直接测试，可以不使用
# 根据不同的组合分为801个文件

cur_path = os.getcwd()
csv_path = os.path.join(cur_path,'util','Data_Entry_2017.csv')


with open(csv_path, mode="r", encoding="utf-8-sig") as data_ori:
    disease={'Atelectasis': 1, 'Cardiomegaly': 2, 'Effusion': 3, 'Infiltration': 4,  'Mass': 5, 
             'Nodule': 6, 'Pneumonia': 7, 'Pneumothorax': 8, 'Consolidation': 9, 'Edema': 10, 
             'Emphysema': 11, 'Fibrosis': 12, 'Pleural_Thickening': 13, 'Hernia': 14, 'No Finding': 0}
    label={}
    label_count=0
    diseases=[]
    reader_ori = csv.reader(data_ori)
    file_train = open(os.path.join(cur_path,'util','train_val_list.txt'), 'r')
    file_test = open(os.path.join(cur_path,'util','test_list.txt'), 'r')
    train_pic=file_train.readline()[:-1]
    test_pic=file_test.readline()[:-1]
    head=next(reader_ori)
    path = os.path.join(cur_path,'dataset','images_001')

    for data in reader_ori:
        patient_dis=data[1].split('|')
        tmp_dis=[]
        for i in patient_dis:
            tmp_dis.append(str(disease[i]))
        tmp_dis.sort(key = lambda x : int(x))
        tmp_dis=','.join(tmp_dis)
        if tmp_dis not in label.keys():
            label[tmp_dis]=label_count
            label_count+=1
            diseases.append([])
        if os.path.exists(os.path.join(path,'images',data[0])):
            pic_path=os.path.join(path,'images',data[0])
        else:
            if path[-1]=='9':
                path=path[:-2]+'10'
            else:
                path=path[:-1]+str(int(path[-1])+1)
            pic_path=os.path.join(path,'images',data[0])
        diseases[label[tmp_dis]].append(pic_path)
    file_train.close()
    file_test.close()
    if not os.path.exists(os.path.join(cur_path,'dataset','diseases')):
        os.mkdir(os.path.join(cur_path,'dataset','diseases'))
    for label,idx in label.items():
        file_disease=open(os.path.join(cur_path,'dataset','diseases',label+'.txt'), 'w+')
        for j in diseases[idx]:
            file_disease.write(j+'\n')
        file_disease.close()