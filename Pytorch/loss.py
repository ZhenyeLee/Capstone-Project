#Ldepth(y, ŷ)
import torch
def l1_criterion(y_pred,y_true):
  #compute point-wise L1 loss
  l_depth = torch.mean(torch.abs(y_pred - y_true))
