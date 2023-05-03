import torch
from densenet import DenseNet
import argparse
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().detach().numpy())

def returnCAM(feature_conv, weight_softmax, idx):
    # 生成CAM图: 输入是feature_conv和weight_softmax 
    bz, nc, h, w = feature_conv.shape  
    # feature_conv和weight_softmax 点乘(.dot)得到cam
    cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w))) 
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    # cam_img = Image.fromarray(cam_img)
    # cam_img = cam_img.resize((224,224))
    cam_img = cv2.resize(cam_img, (224,224))
    return cam_img

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--picture', type=str,
                        default='/Users/xuchengtian/code/chest-diseases-classification/dataset/images_006/images/00011558_008.png',
                        help='choose test picture')
    parser.add_argument('-m','--model',type=str,
                        default='/Users/xuchengtian/code/chest-diseases-classification/models/0427115128/27-11-53-02-epoch1.pth',
                        help='choose test model')
    args = parser.parse_args()
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4722708), (0.22180891))
    ])
    picture = Image.open(args.picture)
    # test_pic = np.array(test_pic)
    test_pic = transform_test(picture)
    test_pic = torch.unsqueeze(test_pic,0)

    net = DenseNet()
    pretrained = torch.load(args.model)
    net.load_state_dict(pretrained['net'])
    net.eval()
    # print(net)
    classify=net.state_dict()['classifier.weight'].squeeze(3).squeeze(2).numpy()
    # print(classify)
    # for name, _ in net.named_modules():
    #     print(name)
    features_blobs=[]
    net._modules['features']._modules.get('relu5').register_forward_hook(hook_feature)
    output = net(test_pic)
    ans = torch.where(output>=0.5,1,0).tolist()[0]
    # print(features_blobs[0].shape)
    img = cv2.imread(args.picture)
    img = cv2.resize(img,(896,896))
    for idx,output in enumerate(ans):
        if output==1:
            CAMs = returnCAM(features_blobs[0], classify, idx)
            heatmap = cv2.applyColorMap(CAMs, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (896,896))
            result = heatmap * 0.3 + img
            cv2.imwrite('./heatmap'+str(idx)+'.jpg', result)
            print('success')
