import os
import argparse
from PIL import Image

import torch
import torchvision
from torchvision import transforms

from model import ConvNet
from datasets import ImageFolder
import cfg


parser = argparse.ArgumentParser(description='PyTorch RPC Predicting')
parser.add_argument('-i', '--image', metavar='PATH', type=str, help='path to image')                    
parser.add_argument('-w', '--model_weight', default=cfg.WEIGHTS_PATH, type=str, metavar='PATH',
                    help='path to latest checkpoint')
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:{}'.format(device))

# classes' name
dataset = ImageFolder(cfg.VAL_PATH)
class_to_idx = dataset.class_to_idx
idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
print(idx_to_class)

# load data
def load_image(imgPath):
    img = Image.open(imgPath).convert('RGB')
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    return trans(img)

# input
img = load_image(args.image)
img = img.unsqueeze(0).to(device)

# create model
model = ConvNet(cfg.NUM_CLASSES).to(device)

# load checkpoint
if args.model_weight:
    if os.path.isfile(args.model_weight):
        print("=> loading checkpoint '{}'".format(args.model_weight))
        checkpoint = torch.load(args.model_weight, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.model_weight))

# predict
model.eval()

with torch.no_grad():
    scores = model(img)
    probability = torch.nn.functional.softmax(scores, dim=1)
    max_value, index = torch.max(probability, 1)

    print('pred_label:{}, pred_class:{}, conf:{:.6f}'.format(
        index.item(), idx_to_class[index.item()], max_value.item()))
