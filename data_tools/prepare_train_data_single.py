import csv
import os

# 数据集生成文件2

# 只筛选出每位患者第一张X光,且必须是患病标签

# 仅仅用于减少训练/测试数据集在cpu上跑看效果

with open("/Users/xuchengtian/code/chest-diseases-classification/util/Data_Entry_2017.csv", mode="r", encoding="utf-8-sig") as data_ori:
    disease={'Atelectasis': 1, 'Cardiomegaly': 2, 'Effusion': 3, 'Infiltration': 4,  'Mass': 5, 
             'Nodule': 6, 'Pneumonia': 7, 'Pneumothorax': 8, 'Consolidation': 9, 'Edema': 10, 
             'Emphysema': 11, 'Fibrosis': 12, 'Pleural_Thickening': 13, 'Hernia': 14, 'No Finding': 0}
    reader_ori = csv.reader(data_ori)
    file_train = open("/Users/xuchengtian/code/chest-diseases-classification/util/train_val_list.txt",'r')
    file_test = open("/Users/xuchengtian/code/chest-diseases-classification/util/test_list.txt",'r')
    train = open("/Users/xuchengtian/code/chest-diseases-classification/dataset/train.txt",'w+')
    test = open("/Users/xuchengtian/code/chest-diseases-classification/dataset/test.txt",'w+')
    train_pic=file_train.readline()[:-1]
    test_pic=file_test.readline()[:-1]
    head=next(reader_ori)
    path='/Users/xuchengtian/code/chest-diseases-classification/dataset/images_001'
    patient=[]
    for data in reader_ori:
        flag=0
        if data[0][0:8] in patient:
            flag=1
        else:
            patient.append(data[0][0:8])

        patient_dis=data[1].split('|')
        tmp_dis=[]
        for i in patient_dis:
            tmp_dis.append(str(disease[i]))
        if tmp_dis[0]=='0':
            flag=1
        tmp_dis=','.join(tmp_dis)
        if os.path.exists(path+'/images/'+data[0]):
            pic_path=path+'/images/'+data[0]
        else:
            if path[-1]=='9':
                path=path[:-2]+'10'
            else:
                path=path[:-1]+str(int(path[-1])+1)
            pic_path=path+'/images/'+data[0]
        if data[0]==train_pic:
            if flag==0:
                train.write(pic_path+' '+tmp_dis+'\n')
            train_pic=file_train.readline()[:-1]
        elif data[0]==test_pic:
            if flag==0:
                test.write(pic_path+' '+tmp_dis+'\n')
            test_pic=file_test.readline()[:-1]
        else:
            print('error in '+data[0]+' '+train_pic+' '+test_pic)
    file_train.close()
    file_test.close()
    train.close()
    test.close()