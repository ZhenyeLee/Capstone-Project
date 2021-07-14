# Capstone-Project
For this project, we want to construct a deep neural network to predict the depth map from a single RGB image in an end-to-end way.The input of the network is RGB images, and after applying the network the output of this system is the corresponding estimated depth maps. The two datasets we are planning to use for training the model, NYU Depth v2 and KITTI, contain RGB images and corresponding ground truth of depth maps. 

## Result
<img width="925" alt="Screen Shot 2021-07-14 at 9 02 54 AM" src="https://user-images.githubusercontent.com/73271404/125635870-d4626f89-d3fe-4b16-bfac-66017923d908.png">





## Data
### 1. NYU Depth v2
The [NYU Depth dataset v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) is composed of 464 indoor scenes with the resolution of 640x480 as recorded by both the RGB and Depth cameras 
from the Microsoft Kinect that can collect ground truth of depth directly. 
<img width="880" alt="Screen Shot 2021-07-13 at 10 03 12 PM" src="https://user-images.githubusercontent.com/73271404/125554394-94186b71-525c-49c1-b61d-4cb2dac74918.png">



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
The code sample written by myself for peer review:
`python train.py`

If you want to run the program, just run `python run_MonocularDepth.ipynb` on Colab that can directly download small NYU v2 dataset. 

### Code Structure ###
`python data.py` reads and pre-processes the NYU v2 dataset.
`python loss.py` contains loss functions.
`python model.py` contains an encoder-decoder model for monocular depth estimation.
`python run_MonocularDepth.ipynb` can be run directly on colab, and it will automatically download the dataset.


## Small test dataset
The small test dataset for NYU-Depth-V2 can be downloaded [here](https://drive.google.com/file/d/1HFAsEQCDUx0UC63Yv5uKE2Z5Z9cKDMV0/view?usp=sharing).
