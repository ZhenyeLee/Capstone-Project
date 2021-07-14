import torch
import torch.nn.functional as func
from math import exp

#Ldepth(y, ŷ)
def l1_criterion(y_pred,y_true):
  #compute point-wise L1 loss
  l_depth = torch.mean(torch.abs(y_pred - y_true))
  
#Lgrad(y, ŷ)
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
def gaussian(window_size, sigma):
  # Calculate the one-dimensional Gaussian distribution vector
  def gauss(point):
    return -(point - (window_size // 2))**2 / float(2 * sigma**2)
  gauss = torch.Tensor([exp(gauss(point)) for point in range(window_size)])
  return gauss / gauss.sum()

def create_window(window_size, channel=1)
  # Create a Gaussian kernel, obtained by matrix multiplication of two one-dimensional Gaussian distribution vectors
  gaussian_kernel1d = gaussian(window_size, 1.5).unsqueeze(1)
  gaussian_kernel2d = gaussian_kernel1d.mm(gaussian_kernel1d.t()).float().unsqueeze(0).unsqueeze(0)
  window = gaussian_kernel2d.expand(channel, 1, window_size, window_size).contiguous()
  return window

def ssim(y_pred, y_true, data_range=None, window=None, size_average=True):
  #If the data is not provided use max and min value from image to calculate data range
  if data_range is None:
    if torch.max(y_pred) > 128:
      max_point = 255
    else:
      max_point = 1
      
    if torch.min(y_pred) < -0.5:
      min_point = -1
    else:
      min_point = 0
    L = max_point - min_point
  else:
    L = data_range

    #get parameter from image
    (_, channel, height, width) = y_pred.size()
    #window_size is 11 by default
    default_window_size=11

    if window is None:
        realWindowsize = min(default_window_size, height, width)
        window = create_window(realWindowsize, channel=channel).to(y_pred.device)
        
    #The formula Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y] is used when calculating variance and covariance .    
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
      
def depth_loss(y_pred, y_true):
  #Ldepth(y, ŷ)
  l_depth = l1_criterion(y_pred, y_true)
  #Lgrad(y, ŷ)
  dx_true, dy_true = image_gradients(y_true)
  dx_pred, dy_pred = image_gradients(y_pred)
  l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))
  #LSSIM (y, ŷ)
  l_ssim = torch.clamp((1 - ssim(y_pred, y_true)) * 0.5, 0, 1)

  #loss 
  w1 = 0.1
  w2 = 1.0
  w3 = 1.0
  loss= (w1 * l_depth) + (w2 * l_edges) + (w3 * l_ssim)
  return loss
