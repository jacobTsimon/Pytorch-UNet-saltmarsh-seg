import matplotlib.pyplot as plt
import numpy as np

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def onehot_to_rgb(oneHot, colorDict, pred = False):
    print(oneHot.size())
    if pred:
        oneHot = oneHot[0:1,:,:,:]
       # print("pred",oneHot.size())
    else:
        oneHot = oneHot[0:1, :, :]
        #print("true",oneHot.size())

    oneHot = np.array(oneHot.detach()) ## ensuring numpy array; should be dims: (1, classes, height, width)
    if pred:
        oneHot = np.argmax(oneHot, axis=1)  ## argmax along second dimension; reduces size to (1, height, width)


    output = np.zeros(oneHot.shape[1:3]+(3,)) ## size of height, width channel, + 3 for the rgb color space
    oneHot= np.transpose(oneHot, [1, 2, 0]) ## ensuring correct dimensions are compared

    for index, color in enumerate(colorDict.keys()):
        if index < len(colorDict.keys()):
            output[np.all(oneHot== index, axis=-1)] = color

    return np.uint8(output)