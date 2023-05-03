from densenet import DenseNet
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

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
    test_pic = Image.open(args.picture)
    # test_pic = np.array(test_pic)
    test_pic = transform_test(test_pic)
    test_pic = torch.unsqueeze(test_pic,0)

    net = DenseNet()
    pretrained = torch.load(args.model)
    net.load_state_dict(pretrained['net'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    output = net(test_pic)
    print(output)
    ans = torch.where(output>=0.5,1,0)
    print(ans)

