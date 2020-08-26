import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import PIL.Image as Image
import os
import Testing as testNet
import Network as net

#####################Function############################
def printImage(img):
    img = img.detach().cpu()
    img = torchvision.utils.make_grid(img)
    img = np.transpose(img, (1,2,0))
    img = img*0.5 + 0.5
    plt.imshow(img)
    plt.show()
#################Hyper Parameter#########################
img_path ='C:/Datasets/OverfittingTest'
fname='1803151818-00000048.jpg'
fname2 = '1803151818-00000048.png'
#net = testNet.BasicNet()
net = testNet.TestNet_Pool()
#####################Etc#################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5, ))
])
l1_loss = nn.L1Loss()
if __name__ =='__main__':
    img = Image.open(os.path.join(img_path, fname))
    img = img.resize((256, 256))
    gt = Image.open((os.path.join(img_path, fname2))).resize((256, 256))#img.copy()
    img = transform(img).to(device)
    img = img.unsqueeze(0)
    gt = transform(gt).to(device)
    gt = gt.unsqueeze(0)
    print(gt.shape)
    printImage(gt)
    net.to(device)
    optimizer = optim.Adam(params=net.parameters(), lr=0.002)

    epoch = 1000

    for e in range(epoch):
        result, temp = net(img)
        loss = l1_loss(result, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e%100 ==99:
            print("e: %d, loss: %f"%(e, loss))
            printImage(img)
            printImage(temp)
            printImage(result)
    #print(l1_loss(img, gt))
