import os
import shutil
import re

imdir = '/home/hopkinsonlab/Documents/ML/Segmentation/marsh_plants/Pytorch-UNet-saltmarsh-seg/Pytorch-UNet-saltmarsh-seg/data/trainimgs_full'
maskdir = '/home/hopkinsonlab/Documents/ML/Segmentation/marsh_plants/Pytorch-UNet-saltmarsh-seg/Pytorch-UNet-saltmarsh-seg/data/masks_full'
#
# nudir = '/home/hopkinsonlab/Documents/ML/Segmentation/marsh_plants/Pytorch-UNet-saltmarsh-seg/Pytorch-UNet-saltmarsh-seg/data/train_imgs'
#
if not os.path.isdir(imdir):
    os.mkdir(imdir)
if not os.path.isdir(maskdir):
    os.mkdir(maskdir)
#
# for dirname, dir, files in os.walk(maskdir):
#     for file in files:
#
#         imname = file.replace('_mask.png','.jpg')
#         imsrc = imdir + '/' + imname
#         nusrc = nudir + '/' + imname
#
#         try:
#             shutil.copy(imsrc,nudir)
#         except:
#             print("copy failed for: {}".format(imname))


topdir = '/home/hopkinsonlab/Documents/ML/Segmentation/marsh_plants/Label_data/2020_Jayant_BH/'

maskre = re.compile('.*(_mask)')

for dirname,dir, files in os.walk(topdir):
    for file in files:
        if re.search(maskre,file):
            try:
                filepath = os.path.join(dirname,file)
                shutil.copy(filepath,maskdir)
            except:
                print("mask {} copy failed...".format(file))

            impath = filepath.replace('_mask.png','.jpg')
            #imsrc = dirname + '/' + imname
            try:
                shutil.copy(impath, imdir)
            except:
                print('image {} copy failed...'.format(impath))
