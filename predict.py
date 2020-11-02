import numpy as np
from unet import UNet
import torch
import os
import cv2 as cv
import matplotlib.pyplot as plt

url = '/home/siat/disk1/Projects/cell/data/test/Fluo-C2DL-MSC/01'

net = UNet(n_channels=1, n_classes=1)
net.load_state_dict(torch.load('320_rotateepoch50.pth'))
net.cuda()

net1 = UNet(n_channels=1, n_classes=1)
net1.load_state_dict(torch.load('/home/siat/disk1/Projects/cell/p2_multiple/default_2epoch20.pth'))
net1.cuda()

with torch.no_grad():
    for u in os.listdir(url):
        i = cv.imread(url+'/'+u, -1)
        i = cv.resize(i, (320, 320))
        i = torch.Tensor((i - np.mean(i)) / (np.max(i) - np.min(i))).unsqueeze(0)
        i = i.unsqueeze(0)
        X = i.cuda()
        pred = net(X)
        pred = pred.sigmoid().cpu().numpy()
        plt.imshow(pred.squeeze(), cmap='gray')
        break
        

##### test #####
with torch.no_grad():
    test = torch.ones([1, 1, 320, 320])
    test[0, 0, 100] = 0
    test_res = net(test.cuda()).sigmoid().cpu().numpy().squeeze()
    plt.imshow(test_res, cmap='gray')
