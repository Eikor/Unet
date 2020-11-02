import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from unet import UNet
from torch.utils.tensorboard import SummaryWriter
from dataset import CellDataset
from loss import Loss
from torch.utils.data import DataLoader, random_split


torch.cuda.empty_cache()
############ Training Info ############
url_train = "/home/siat/disk1/Projects/cell/data/train"
url_test = "/home/siat/disk1/Projects/cell/data/test"
crop_size = 512
validation_split = 0.2
batch_size = 8
lr = 0.01
epochs = 50
UsingDataSet = ['Fluo-C2DL-MSC', 
                # 'DIC-C2DH-HeLa',
                # 'Fluo-N2DH-GOWT1', 
                # 'Fluo-N2DH-SIM+',
                # 'Fluo-N2DL-HeLa',
                # 'PhC-C2DH-U373'
                ]

dataset = CellDataset(url_train, UsingDataSet, crop_size)
n_val = int(len(dataset) * validation_split)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
###################################


############ load module ############
net = UNet(1, 1)
net.cuda()
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
criterion = Loss(0.5, 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
#####################################


for epoch in range(epochs):
    net.train()
    epoch_loss = 0
    global_step = 0
    print("epoch: {} ".format(epoch))

    for batch in train_loader:
        imgs = batch['image'].cuda()
        true_masks = batch['mask'].cuda()

        masks_pred = net(imgs)
        loss = criterion(masks_pred, true_masks)
        print("loss: {:.5f}, dice loss: {:.5f}".format(loss[0], loss[1]))
        loss = 0.8*loss[0]+0.2*loss[1]
        # writer.add_scalar('Loss/train', loss.item(), global_step)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()

        
        global_step += 1
        
        # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        # writer.add_images('images', imgs, global_step)
        # writer.add_images('masks/true', true_masks, global_step)
        # writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        
    ###### validation ######
    # torch.cuda.empty_cache()
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image']
            label = batch['mask']
            imgs = imgs.cuda()
            label = label.cuda()
            pred = net(imgs)
            val_loss += np.sum(criterion(pred, label))/len(val_loader)
        scheduler.step(val_loss)
    print('Validation: {}'.format(val_loss))
    ########################

    # writer.add_scalar('test', val_score, epoch)
torch.save(net.state_dict(), '320_rotate' + f'epoch{epoch + 1}.pth')



