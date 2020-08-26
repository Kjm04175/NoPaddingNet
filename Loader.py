import torch
import torchvision
import torchvision.transforms as transforms
import os
import PIL.Image as Image
import random
import cv2
from torch.utils.data import DataLoader
import  numpy as np
from skimage.color import rgb2lab, lab2rgb

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

class ImageLoader(DataLoader):
    def __init__(self, path = '/temp',img_shape=(540,540)):
        self.path = path
        width, height = img_shape
        self.hr_transform = transforms.Compose([
            transforms.Resize((width//4*4,height//4*4),Image.BICUBIC),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((width//4, height//4), Image.BICUBIC),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
        ])
        self.img_list = os.listdir(path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        fname = self.img_list[idx]
        img = Image.open(os.path.join(self.path, fname))

        hr_img = self.hr_transform(img)
        small_img = self.lr_transform(img)
        return hr_img.float(), small_img.float()

class ImageLoader_cv(DataLoader):
    def __init__(self, path = '/temp', transform = None):
        self.path = path
        self.transform = transform
        self.img_list = os.listdir(path)


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        fname = self.img_list[idx]
        img = cv2.imread(os.path.join(self.path, fname))
        w, h = img.width
        w = w//8*8
        h = h//8*8
        img = img.resize((w, h))
        randval = random.randrange(2,4)
        small_img = img.resize((int(w/randval), int(h/randval)))
        small_img = small_img.resize((w, h))
        if self.transform is not None:
            img = self.transform(img)
            small_img = self.transform(small_img)
        return img, small_img