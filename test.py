from utils.utils import csv_writer, prediction
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet
import torch.nn as nn
import argparse
import torch
import cv2
import os

models = {'resnet18':512, 'resnet34':512, 'resnet50':2048, 'resnet101':2048}
model_names = list(models.keys())
parser = argparse.ArgumentParser(description='Propert ResNets for mini-nico in pytorch')
parser.add_argument('--ckpt',
                    help='The directory used to test the trained models',
                    default=None, type=str)
parser.add_argument('--test_img_path',
                    help='The directory is where test imgs in',
                    default="./data/test/", type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--gpu', dest='gpu', action='store_true',
                    help='whether choose the gpu to train')                    

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

class test_transform(object):
    def __init__(self):
        super().__init__()
    def __call__(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        return transform(img)

def set_model(args):
    model = resnet.__dict__[args.arch]()
    model.fc = nn.Linear(in_features=models[args.arch], out_features=5)
    if args.gpu:
        model.cuda()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)

    return model

def main():
    args = parser.parse_args()
    model = set_model(args)
    test(model, args)

def test(model, args):
    """
    Run test
    """
    # switch to evaluate mode
    model.eval()
    img_label = {}
    count = 1
    with torch.no_grad():
        imgs = os.listdir(args.test_img_path)
        imgs.sort(key=lambda x : int(x.split('.')[0]))
        for img in imgs:
            image = cv2.imread(args.test_img_path + img)
            input_var = test_transform()(image)
            input_var = input_var.unsqueeze(0)
          
            # compute output
            if args.gpu:
                input_var = input_var.cuda()
            output = model(input_var)
            output = output.float()
            predict_result = prediction(output)
            img_label[img.split('.')[0]] = predict_result
            print(str(count) + "___Finishing predicting the " + img + ", its predict_result is " + str(predict_result) + " !!!")
            count += 1
    # writing in csv
    header = ('test_img_id', 'animal_class_id')
    csv_writer(img_label , header)

if __name__ == '__main__':
    main()