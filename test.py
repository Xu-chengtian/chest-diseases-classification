from densenet import DenseNet
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--picture', type=str,
                        default='/Users/xuchengtian/code/chest-diseases-classification/dataset/images_005/images',
                        help='choose test picture')
    parser.add_argument('-m','--model',type=str,
                        default='/Users/xuchengtian/code/chest-diseases-classification/models/model.pth',
                        help='choose test model')
    parser.add_argument('-s','--show', action= "store_true",help='test all picture')
    args = parser.parse_args()
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4722708), (0.22180891))
    ])
    # test_pic = Image.open(args.picture)
    # # test_pic = np.array(test_pic)
    # test_pic = transform_test(test_pic)
    # test_pic = torch.unsqueeze(test_pic,0)

    # net = DenseNet()
    # pretrained = torch.load(args.model)
    # net.load_state_dict(pretrained['net'])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net.to(device)
    # output = net(test_pic)
    # print(output)
    # ans = torch.where(output>=0.5,1,0)
    # print(ans)
    if args.show:
        net = DenseNet()
        pretrained = torch.load(args.model, map_location='cpu')
        net.load_state_dict(pretrained['net'])
        net.eval()
        files = os.listdir(os.path.join(os.getcwd(),'dataset','diseases'))
        for file in files:
            if len(file)<7:
                continue
            print(file)
            pic_file = open(os.path.join(os.getcwd(),'dataset','diseases',file))
            picture_path = pic_file.readline()[:-1]
            while(picture_path):
                picture = Image.open(picture_path)
                picture = transform_test(picture)
                picture = torch.unsqueeze(picture,0)
                output = net(picture)
                ans = torch.where(output>=0.5,1,0).tolist()[0]
                if 1 in ans:
                    print(ans)
                    print(picture_path)

                picture_path = pic_file.readline()[:-1]
    else:
        for pic_path in os.listdir(args.picture):
            picture_path=os.path.join(args.picture,pic_path)
            test_pic = Image.open(picture_path)
            # test_pic = np.array(test_pic)
            test_pic = transform_test(test_pic)
            test_pic = torch.unsqueeze(test_pic,0)

            net = DenseNet()
            pretrained = torch.load(args.model, map_location='cpu')
            net.load_state_dict(pretrained['net'])
            net.eval()
            output = net(test_pic)
            ans = torch.where(output>=0.5,1,0).tolist()[0]
            if 1 in ans:
                print(ans)
                print(picture_path)
