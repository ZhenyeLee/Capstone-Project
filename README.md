# Capstone-Project
For this project, we want to construct a deep neural network to predict the depth map from a single RGB image in an end-to-end way.The input of the network is RGB images, and after applying the network the output of this system is the corresponding estimated depth maps. The two datasets we are planning to use for training the model, NYU Depth v2 and KITTI, contain RGB images and corresponding ground truth of depth maps. 

## Datasets
### 1. NYU Depth v2
The [NYU Depth dataset v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) is composed of 464 indoor scenes with the resolution of 640x480 as recorded by both the RGB and Depth cameras 
from the Microsoft Kinect that can collect ground truth of depth directly. 

### 2. KITTI
The [KITTI](http://www.cvlibs.net/datasets/kitti/) is a large dataset that is composed of 56 outdoor scenes including the "city", "residential" categories of the raw data, and so on. 

## Loss Function
 The total loss Function are the weighted sum of three loss functions:
 #### 1.Ldepth(y, ŷ) compute point-wise L1 loss
 #### 2.Lgrad(y, ŷ) compute image gradient loss
 #### 3.LSSIM (y, ŷ)(Structural similarity index) compute the similarity of two images
  Reference : https://ece.uwaterloo.ca/~z70wang/research/ssim/ 
  
  #### Algorithm : <img width="1165" alt="Screen Shot 2021-07-13 at 12 58 00 PM" src="https://user-images.githubusercontent.com/73271404/125504126-4bcbc3ae-b7d3-40e1-b987-989124c17683.png">
