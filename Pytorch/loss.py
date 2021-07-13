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

#LSSIM (y, ŷ)
#Structural similarity index is a method for predicting similarity of two images
#An image quality metric that assesses the visual impact of three characteristics of an image: luminance, contrast and structure.
from math import exp
def gaussian(window_size, sigma):
  #create gaussian filter 
  gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
  return gauss/gauss.sum()

def create_window(window_size, channel=1):
  _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
  _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
  window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
  return window

def ssim(y_pred, y_true, data_range=None, window=None, size_average=True):
  #Data range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
  #If the data is not provided use max and min value from image to calculate data range
  if data_range is None:
    if torch.max(y_pred) > 128:
      max_y_pred = 255
    else:
      max_y_pred = 1
    
    if torch.min(y_pred) < -0.5:
      min_y_pred = -1
    else:
     min_y_pred = 0
    L = max_val - min_val
  else:
    L = data_range

    #get parameter from image
    (_, channel, height, width) = y_pred.size()
    #window_size is 11 by default
    default_window_size=11

    if window is None:
        realWindowsize = min(default_window_size, height, width)
        window = create_window(realWindowsize, channel=channel).to(y_pred.device)

    #mu_x the average of x
    mu_x = func.conv2d(y_pred, window, padding=0, groups=channel)
    #mu_y the average of y
    mu_y = func.conv2d(y_true, window, padding=0, groups=channel)
    #sigma_xy the covariance of x and y
    Sigma_xy = func.conv2d(y_pred * y_true, window, padding=0, groups=channel) - (mu_x * mu_x) * (Uy * mu_y)

    #K1 = 0.01 and k2 = 0.03 by default
    K1 = 0.01
    K2 = 0.03
    #C1 and C2 two variables to stabilize the division with weak denominator
    ##L is the dynamic range of the pixel-values which either provided by user or calculate from before
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    Denominator = (2 * mu_x * mu_y + C1) * (2 * Sigma_xy + C2)
    #sigma_x^2 is the variance of x
    Sigma_x_sq = func.conv2d(y_pred * y_pred, window, padding=0, groups=channel) - mu_x * mu_x
    #sigma_y^2 is the variance of y
    Sigma_y_sq = func.conv2d(y_true * y_true, window, padding=0, groups=channel) - mu_y * mu_y

    Numerator = (mu_x * mu_x + C1) *(Sigma_x_sq + Sigma_y_sq + C2)

    ssim_map = Denominator / Numerator

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
