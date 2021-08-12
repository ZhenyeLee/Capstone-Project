<h1 align="center">Monocular Depth Estimation Using a Deep Network</h1>

<p>For this project, we constructed a deep neural network to predict the depth map from a single RGB image in an end-to-end way. The input of the network is RGB images, and after applying the network the output of this system is the corresponding estimated depth maps. The two datasets we are planning to use for training the model, NYU Depth v2 and KITTI, contain RGB images and corresponding ground truth of depth maps. </p>

## Result
<h3 align="center"> Demo for indoor scene in Engineering Hall, UW Madison</h3>
<p align="center">
<img src="https://user-images.githubusercontent.com/73271404/128800359-ddf6a52e-6075-4bae-9edb-e2d47b60dd74.gif" width="400"/><img src="https://user-images.githubusercontent.com/73271404/128800367-8877c642-a4e7-4abf-92ca-cf40a32c5fd5.gif" width="400"/>
 </p>
 
 <h3 align="center"> Demo for outdoor scene at State Street, Madison</h3>
<p align="center">
<img src="https://user-images.githubusercontent.com/73271404/128801153-f608a47d-9adf-4cd3-939c-5baa7ed98c2e.gif"/>
 <img src="https://user-images.githubusercontent.com/73271404/128801156-e4efe4ca-a0e0-4a48-a8ee-4aa7f08a972f.gif"/>
 </p>


## DataSet
### 1. NYU Depth v2
The [NYU Depth dataset v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) is composed of 464 indoor scenes with the resolution of 640x480 as recorded by both the RGB and Depth cameras 
from the Microsoft Kinect that can collect ground truth of depth directly. 


### 2. KITTI
The [KITTI](http://www.cvlibs.net/datasets/kitti/) is a large dataset that is composed of 56 outdoor scenes including the "city", "residential" categories of the raw data, and so on. 

## Loss Function
 The total loss Function are the weighted sum of three loss functions:
 #### 1. Ldepth(y, ŷ) compute point-wise L1 loss
 #### 2. Lgrad(y, ŷ) compute image gradient loss
 #### 3. LSSIM (y, ŷ)(Structural similarity index) compute the similarity of two images
  Reference : https://ece.uwaterloo.ca/~z70wang/research/ssim/ 
  
  #### [Algorithm](https://en.wikipedia.org/wiki/Structural_similarity#Algorithm) : <img width="1165" alt="Screen Shot 2021-07-13 at 12 58 00 PM" src="https://user-images.githubusercontent.com/73271404/125504126-4bcbc3ae-b7d3-40e1-b987-989124c17683.png">


## Usage
If you want to run the whole program, just run one of `.ipynb` files on Colab that can directly download the entire NYU depth v2 dataset or the KITTI dataset. 

### Code Structure ###
#### NYU depth v2 Dataset ####
`NYU/Initial_Model/ train.py` trains a model for the NYU v2 dataset.  
`NYU/Initial_Model/ data.py` reads and pre-processes the NYU v2 dataset.  
`NYU/Initial_Model/ loss.py` contains loss functions.  
`NYU/Initial_Model/ model.py` contains an encoder-decoder model for monocular depth estimation. This part is from model.py that can be download from https://github.com/ialhashim/DenseDepth/tree/master/PyTorch.  
  
  
`NYU/NYU_Standard_MonocularDepth.ipynb` can be run directly on colab, and it will automatically download the entire NYU depth v2 dataset. The original encoder-decoder architecture is used in this `.ipynb` file.  
`NYU/NYU_addbatch_MonocularDepth.ipynb` This is the modified encoder-decoder architecture adding BatchNormalization layer.  
`NYU/NYU_addup_MonocularDepth.ipynb` This is the modified encoder-decoder architecture adding one more 2x upsampling layer.  
`NYU/NYU_simpledecoder_MonocularDepth.ipynb` This is the modified encoder-decoder architecture without skip-connections.  

#### KITTI Dataset ####
`KITTI/kitti_equalloss_MonocularDepth.ipynb` can be run directly on colab, and it will automatically download the KITTI dataset. This is the original encoder-decoder architecture with the equal-weighted loss function. In this `.ipynb` file, the modular of data reading is rewritten for the KITTI dataset.


## Small test dataset
The small test dataset for NYU-Depth-V2 can be downloaded [here](https://drive.google.com/file/d/1HFAsEQCDUx0UC63Yv5uKE2Z5Z9cKDMV0/view?usp=sharing).
