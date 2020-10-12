import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from skimage import io
# import PIL.Image as Image
from PIL import Image as Image

def ZeroCenter(img):

    for i in range(3):
        mu = np.ones_like(img[i]) * img[i].mean()
        img[i] = img[i] - mu

    return img


# root_dir = 'C:/Users/or8be/OneDrive/Desktop/Electrical Engineering B.Sc/Deep Learning/Final Project/Custom_Data'
# csv_file = 'C:/Users/or8be/OneDrive/Desktop/Electrical Engineering B.Sc/Deep Learning/Final Project/csv_style.csv'


class PaintingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file, encoding='latin1')   # array of paths & labels
        # self.annotations = open(csv_file,"r").read()
        self.root_dir = root_dir                   # path to dataset directory
        self.transforms = transforms               # transform

    def __len__(self):
        return len(self.annotations)               # num of images

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 0]
        # print(img_path)            # check corrupt files XXXXXXXXX
        image = io.imread(img_path)                                   # load image from source
        label = torch.tensor(int(self.annotations.iloc[index, 1]))  # get label from file as tensor

        # image = ZeroCenter(pic)                                     # zero-centering the image

        if self.transforms:
            image = self.transforms(image)

        return image, label
