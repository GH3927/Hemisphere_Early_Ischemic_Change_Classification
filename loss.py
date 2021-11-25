import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class Weighted_Binary_Cross_Entropy(nn.Module):
    def __init__(self, weights=None):
        super(Weighted_Binary_Cross_Entropy, self).__init__()
        self.weights = weights
    
    def forward(self, inputs, targets):    
        if self.weights is not None:
            assert len(self.weights) == 2
            
            loss = self.weights[1] * (targets * torch.log(inputs)) + \
                   self.weights[0] * ((1 - targets) * torch.log(1 - inputs))
        else:
            loss = targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs)
    
        return torch.neg(torch.mean(loss))