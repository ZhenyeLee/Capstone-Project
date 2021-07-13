#Ldepth(y, ŷ)
import torch
def l1_criterion(y_pred,y_true):
  #compute point-wise L1 loss
  l_depth = torch.mean(torch.abs(y_pred - y_true))
  
#Lgrad(y, ŷ)
import torch.nn.functional as func
def image_gradients(image):
  #compute image gradient loss
  left = image
  right = func.pad(image,[0, 1, 0, 0])[:, :, :, 1:]
  top = image
  bottom = func.pad(image, [0, 0, 0, 1])[:, :, 1:, :]

  dx = right - left
  dy = bottom - top
  dx[:, :, :, -1] = 0
  dy[:, :, -1, :] = 0

  return dx,dy
