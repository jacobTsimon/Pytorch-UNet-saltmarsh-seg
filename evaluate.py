import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

#new metrics and code for creating + saving confusion matrices
from utils.metrics import save_confusion_matrix



#class names - made to match myinfo.py from deeplab
class_names = ['Sarcocornia','Batis','deadSpartina','Spartina','Juncus','Borrichia','Limonium',"Other",'background']
@torch.inference_mode()
def evaluate(net, dataloader, device, amp,exp = None, newmetrics = None,set = "val"):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            #add batch into new metrics
            newmetrics.add_batch(mask_true.cpu().numpy(),mask_pred.argmax(axis=1).cpu().numpy())

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #log confusion matrix
    Acc = newmetrics.Pixel_Accuracy()
    Acc_class = newmetrics.Pixel_Accuracy_Class()
    mIoU, IoU = newmetrics.Mean_Intersection_over_Union()
    FWIoU = newmetrics.Frequency_Weighted_Intersection_over_Union()
    DICE = dice_score / max(num_val_batches, 1)

    confmat = save_confusion_matrix(newmetrics.confusion_matrix, class_names, normalize=True,
                                    file_name='{}_confusion_matrix.jpg'.format(set))

    results = {"ACC": Acc,"ACC_CLASS":Acc_class,"MIOU":mIoU,"IOU":IoU,"FWIOU":FWIoU,"DICE":DICE,"CONFMAT":confmat}



    net.train()
    return results
