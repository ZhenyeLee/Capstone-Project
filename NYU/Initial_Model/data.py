import numpy as np
import torch
import random

from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
from zipfile import ZipFile

# the following block, loadZipFile, is from the block named loadZipToMem in data.py
# that can be download from https://github.com/ialhashim/DenseDepth/tree/master/PyTorch.
def loadZipFile(zip_file):
    input_zip = ZipFile('nyu_data.zip')
    # extract a file in memory
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    # decode the bytes to str while putting one RGB image name and corresponding depth name to one row, and the split them.
    train_names = list(row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\r\n')if len(row) > 0)
    test_names = list(row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n')if len(row) > 0)
    return data, train_names, test_names



class NYU_TrainAugmentDataset(Dataset):
    def __init__(self, data, nyu2, transform=None):
        self.data, self.nyu_dataset = data, nyu2
        # the depth range from 0.1m to 10m.
        self.minDepth = 10
        self.maxDepth = 1000

    def __getitem__(self, idx):
        # sample is a list containing names of a RGB image and corresponding depth image
        sample = self.nyu_dataset[idx]
        # PIL.Image.open() Opens and identifies the given image file.
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )
        # Augmentation: random horizontal flip
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        # resize the depth ground truth from (640,480) to (320,240) that is consist with the size of the predicted output.
        depth = depth.resize((320, 240))
        
        # go from Numpy’s arrays to PyTorch’s tensors
        # reshape to (H x W x C)
        # normalize image and depth to [0,1], and then multiply all elements in depth by 1000
        # put all elements in RGB image into the range [0, 1]; put all elements in depth into the range [10, 1000]
        # change the dimensions of the tensor to (C x H x W) that meets the size requirement of input in CNN.
        image = torch.clamp(torch.from_numpy(np.array(image).reshape(480,640,3)).float()/255,0,1).permute(2, 0, 1)
        depth = torch.clamp(torch.from_numpy(np.array(depth).reshape(240,320,1)).float()/255*self.maxDepth, self.minDepth, self.maxDepth).permute(2, 0, 1)
        # depth normalization
        depth = self.maxDepth/depth
        sample = {'image': image, 'depth': depth}
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class NYU_TestDataset(Dataset):
    def __init__(self, data, nyu2, transform=None):
        self.data, self.nyu_dataset = data, nyu2
        # the depth range from 0.1m to 10m.
        self.minDepth = 10
        self.maxDepth = 1000

    def __getitem__(self, idx):
        # sample is a list containing names of a RGB image and corresponding depth image
        sample = self.nyu_dataset[idx]
        # PIL.Image.open() Opens and identifies the given image file.
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )
        # resize the depth ground truth from (640,480) to (320,240)
        depth = depth.resize((320, 240))
    
        # go from Numpy’s arrays to PyTorch’s tensors
        # reshape to (W x H x W)
        # normalize image and depth to [0,1], and then divide all elements in depth by 10
        # put all elements in RGB image into the range [0, 1]; put all elements in depth into the range [10, 1000]
        # change the dimensions of the tensor to (C x H x W) that meets the size requirement of input in CNN.
        image = torch.clamp(torch.from_numpy(np.asarray(image).reshape(480,640,3)).float()/255,0,1).permute(2, 0, 1)
        depth = torch.clamp(torch.from_numpy(np.asarray(depth).reshape(240,320,1)).float()/10, self.minDepth, self.maxDepth).permute(2, 0, 1)
        # depth normalization
        depth = self.maxDepth/depth
        sample = {'image': image, 'depth': depth}
        return sample

    def __len__(self):
        return len(self.nyu_dataset)
