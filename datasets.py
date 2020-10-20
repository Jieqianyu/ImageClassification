import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os

class myDataset(torch.utils.data.Dataset):
    def __init__(self, root, label, mode='train'):
        super().__init__()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif mode == 'test':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        self.root = root
        with open(label, 'r') as f:
            self.imgs_info = f.readlines()

    def __getitem__(self, idx):
        img_info = self.imgs_info[idx].replace('\n', '')
        elem = img_info.split(' ')
        img_path = os.path.join(self.root, elem[0])
        label = int(elem[1])

        # read image
        image = self.transform(Image.open(img_path))

        return image, label

    def __len__(self):
        return len(self.imgs_info)

def ImageFolder(root, mode='train'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif mode == 'test':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
           
    return torchvision.datasets.ImageFolder(root, transform)