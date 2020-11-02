import torch
from torch.autograd import Function
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, alpha, epsilon):
        super(Loss, self).__init__()
        self.al = alpha
        self.ep = epsilon
        # self.wbce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, label):
        pred = torch.sigmoid(pred).clamp(0.001, 0.999)
        
        # scale = torch.sum(label)/label.numel()
        # weight = torch.ones_like(pred) * scale
        # weight = weight + label * (1 - 2*torch.sum(label)/label.numel())
        # weight = 1 - 2*torch.sum(label)/label.numel()

        ### focal weight ###
        gamma = 1
        weight = label * (1 - pred) + (1 - label) * pred
        weight = torch.pow(weight, gamma)
        ####################

        dice = 1 - (2*torch.sum(label*pred) + self.ep) / (torch.sum(label) + torch.sum(pred) + self.ep)
        loss = -torch.sum(weight * ((label * pred.log()) + (1 - label) * (1 - pred).log()))/pred.numel()
        # print("loss: {:.5f}, dice loss: {:.5f}".format(loss, dice))
        # loss = self.al*loss + (1-self.al)*dice
        
        return [loss, dice]





if __name__ == "__main__":
    pred = torch.ones([1, 5, 5])
    label = torch.zeros_like(pred)
    label[0, 1] = 1
