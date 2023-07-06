
import os
os.getcwd()

# import general_module which locate at my parent directory's child
import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
import torch.nn as nn
from general_module.training import *

# other
def getPosWeight(lbl):
    lbl = lbl.tolist()
    lbl = torch.Tensor(lbl)
    pos_weight  = (len(lbl) - lbl.sum())/ lbl.sum()
    if pos_weight.isinf() or lbl.sum() == len(lbl):
        pos_weight= torch.Tensor([0])
    pos_weight = best_device(pos_weight)
    return pos_weight

def getItemWeight(lbl):
    squeeze = []
    p=0
    for i in lbl:
      i= int(i)
      squeeze.append(i)
      p=p+i

    l = len(squeeze)
    n = l-p

    ps = (l-p)/l
    ns = (l-n)/l

    return (ps,ns)

def list_weight(labels,weight):
    weight_list = []
    for l in labels:

        if l.item() ==0:
            weight_list.append([weight[1]])
        else:
            weight_list.append([weight[0]])
    weight_list = torch.tensor(weight_list)
    return best_device(weight_list)



# Focal Loss
class CFBCE():
    def __init__(self, gamma=2, alpha=0.75):
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        lloss= self.alpha*(torch.pow(1-(-BCE_loss).exp(),self.gamma)*(BCE_loss))
        return lloss.mean()

class WFBCEb():
    def __init__(self, gamma=2):
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        self.weight = getItemWeight(targets)
        self.alpha = list_weight(labels = targets, weight=self.weight)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        lloss= self.alpha*(torch.pow(1-(-BCE_loss).exp(),self.gamma)*(BCE_loss))
        return lloss.mean()

# Sample
class WBCEs():
    def __init__(self, lbl):
        self.weight = getItemWeight(lbl)
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        lloss= list_weight(labels = targets, weight=self.weight)*(BCE_loss)
        return lloss.mean()


class WBCEb():
    def __init__(self):
        pass
    
    def forward(self, inputs, targets):
        self.weight = getItemWeight(targets)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        lloss= list_weight(labels = targets, weight=self.weight)*(BCE_loss)
        return lloss.mean()
    

class BBCEs():
    def __init__(self, lbl):
        self.weight = getPosWeight(lbl)
    
    def forward(self, inputs, targets):
        lloss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight = self.weight)
        return lloss.mean()


class BBCEb():
    def __init__(self):
        pass
    
    def forward(self, inputs, targets):
        self.weight = getPosWeight(targets)
        lloss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight = self.weight)
        return lloss.mean()
    
class BCE():
    def __init__(self):
        pass

    def forward(self, inputs, targets):
        lloss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        return lloss.mean()

 

