import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
import logging
import os
from matplotlib import pyplot as plt

class CellDataset(Dataset):
    def __init__(self, dir, dataset, crop_size):
        self.crop_size = crop_size
        self.images = []
        self.masks = []
        for ds in dataset:
            url = os.path.join(dir, ds)
            ## load 01 data
            path = url + '/01'
            for f in os.listdir(path):
                self.images.append(os.path.join(path, f))
                self.masks.append(path+'_ST/SEG/man_seg'+f[1:])
            ## load 02 data
            path = url +'/02'
            for f in os.listdir(path):
                self.images.append(os.path.join(path, f))
                self.masks.append(path+'_ST/SEG/man_seg'+f[1:])
        
    def preprocess(self, img, mask):

        ### rotate ###
        rotate_seed = int((torch.rand(1)*360).item())
        rotate = cv.getRotationMatrix2D((160, 160), rotate_seed, 1)
        img = cv.warpAffine(img, rotate, (320, 320))
        mask = cv.warpAffine(mask, rotate, (320, 320))
        ### crop ###
        # h, w = img.shape
        # crop_seed = torch.rand(2)
        # shift = ((np.array([h, w]) - self.crop_size) * crop_seed.numpy()).astype('int')
        # img = img[shift[0]:shift[0]+self.crop_size, shift[1]:shift[1]+self.crop_size]
        # mask = mask[shift[0]:shift[0]+self.crop_size, shift[1]:shift[1]+self.crop_size]
        ############

        img = np.array(img)
        img = img[np.newaxis, :]
        
        img = (img - np.mean(img)) / (np.max(img) - np.min(img))
        
        label = np.unique(mask)
        for cell in label[1:]:
            iy, ix = np.where((mask == cell) > 0)
            x = np.min(ix)
            y = np.min(iy)
            x2 = np.max(ix)
            y2 = np.max(iy)
            h = x2 - x
            w = y2 - y
        
        
        mask = np.clip(mask, 0, 1).astype('int8')
        mask = mask[np.newaxis, :]
        return img, mask

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        img = cv.imread(self.images[i], -1)
        mask = cv.imread(self.masks[i], -1)
        img, mask = self.preprocess(img, mask)
        
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


if __name__ == "__main__":
    UsingDataSet = ['Fluo-C2DL-MSC', 
                # 'DIC-C2DH-HeLa',
                # 'Fluo-N2DH-GOWT1', 
                # 'Fluo-N2DH-SIM+',
                # 'Fluo-N2DL-HeLa',
                'PhC-C2DH-U373'
                ]
    dataset = CellDataset('/home/siat/disk1/Projects/cell/data/train', UsingDataSet, 512)
    
    fig = plt.figure()
    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['image'].shape, sample['mask'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample['mask'].squeeze(), cmap='gray')

        if i == 3:
            plt.show()
            break












