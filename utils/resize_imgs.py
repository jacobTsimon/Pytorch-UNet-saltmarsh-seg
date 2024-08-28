import numpy as np
import torch
from PIL import Image
import os
import shutil


topdir = '/home/hopkinsonlab/Documents/ML/Segmentation/marsh_plants/Pytorch-UNet-saltmarsh-seg/Pytorch-UNet-saltmarsh-seg/data/trainimgs_full/'
newdir = '/home/hopkinsonlab/Documents/ML/Segmentation/marsh_plants/Pytorch-UNet-saltmarsh-seg/Pytorch-UNet-saltmarsh-seg/data/trainimgs_sized/'
size = (580,360)

if not os.path.isdir(newdir):
    os.mkdir(newdir)


for dirname,dir,files in os.walk(topdir):
    for mask in files:
        if mask == ".keep":
            continue
        print(mask)
        maskpath = os.path.join(dirname,mask)
        im = Image.open(maskpath)
        print(im.size)
        im1 = im.resize(size,resample=Image.NEAREST)
        newpath = os.path.join(newdir,mask)
        im1.save(newpath)
