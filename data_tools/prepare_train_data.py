import csv
import os

# 数据集生成文件
# 训练前必须使用本程序
# 利用util文件中的分类txt，结合图像标签csv文件，生成测试集和训练集文件共后续使用
# 生成文件存放在dataset目录下
# 训练前查看数据集是否完整，不然会报错


cur_path = os.getcwd()
csv_path = os.path.join(cur_path,'util','Data_Entry_2017.csv')


with open(csv_path, mode="r", encoding="utf-8-sig") as data_ori:
    disease={'Atelectasis': 1, 'Cardiomegaly': 2, 'Effusion': 3, 'Infiltration': 4,  'Mass': 5, 
             'Nodule': 6, 'Pneumonia': 7, 'Pneumothorax': 8, 'Consolidation': 9, 'Edema': 10, 
             'Emphysema': 11, 'Fibrosis': 12, 'Pleural_Thickening': 13, 'Hernia': 14, 'No Finding': 0}
    reader_ori = csv.reader(data_ori)
    file_train = open(os.path.join(cur_path,'util','train_val_list.txt'), 'r')
    file_test = open(os.path.join(cur_path,'util','test_list.txt'), 'r')
    train = open(os.path.join(cur_path,'dataset','train.txt'), 'w+')
    test = open(os.path.join(cur_path,'dataset','test.txt'),'w+')
    train_pic=file_train.readline()[:-1]
    test_pic=file_test.readline()[:-1]
    head=next(reader_ori)
    path = os.path.join(cur_path,'dataset','images_001')

    for data in reader_ori:
        patient_dis=data[1].split('|')
        tmp_dis=[]
        for i in patient_dis:
            tmp_dis.append(str(disease[i]))
        tmp_dis=','.join(tmp_dis)
        if os.path.exists(os.path.join(path,'images',data[0])):
            pic_path=os.path.join(path,'images',data[0])
        else:
            if path[-1]=='9':
                path=path[:-2]+'10'
            else:
                path=path[:-1]+str(int(path[-1])+1)
            pic_path=os.path.join(path,'images',data[0])
        if data[0]==train_pic:
            train.write(pic_path+' '+tmp_dis+'\n')
            train_pic=file_train.readline()[:-1]
        elif data[0]==test_pic:
            test.write(pic_path+' '+tmp_dis+'\n')
            test_pic=file_test.readline()[:-1]
        else:
            print('error in '+data[0]+' '+train_pic+' '+test_pic)
    file_train.close()
    file_test.close()
    train.close()
    test.close()
