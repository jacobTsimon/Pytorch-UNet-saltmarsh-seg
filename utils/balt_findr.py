import os
import shutil
from PIL import Image
import numpy as np

maskdir = '/home/hopkinsonlab/Documents/ML/Segmentation/marsh_plants/Pytorch-UNet-saltmarsh-seg/Pytorch-UNet-saltmarsh-seg/data/masks_full/'

testdir = '/home/hopkinsonlab/Documents/ML/Segmentation/marsh_plants/Pytorch-UNet-saltmarsh-seg/Pytorch-UNet-saltmarsh-seg/data/masks_clean/'

if not os.path.isdir(testdir):
    os.mkdir(testdir)

for dir,dirname,files in os.walk(maskdir):
    for mask in files:
        if mask == ".keep":
            continue
        path = os.path.join(dir,mask)

        arr = np.array(Image.open(path))

        print(arr.shape)
        if np.any(arr[:,:,0] == 150):
            print(mask)

            arr[(arr == [150,255,14]).all(-1)] = [255,255,255]

            newpath = os.path.join(testdir,mask)


            newim = Image.fromarray(arr)
            newim.save(newpath)
        else:
            newpath = os.path.join(testdir, mask)
            shutil.copy(path,newpath)

