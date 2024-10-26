import argparse
import logging
import os
import time
import numpy as np
import torch
import math
import torch.nn.functional as F
from glob import glob
import pandas as pds
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from medpy.metric import binary
from scipy.spatial.distance import directed_hausdorff
from utils.load_ISIC import isic_loader
from sklearn.metrics import confusion_matrix

from network.IHE_Net(C_M) import CUM-Net 
from network.IHE_Net(C_T) import CUT-Net 

from torch.utils.data import DataLoader
torch.cuda.set_device(0)

logging.getLogger().setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='best_model.pth', 
                        metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white", default=0.5)
    parser.add_argument('--scale', '-s', type=float,help="Scale factor for the input images",
                        default=0.5)
    parser.add_argument('--num_classes', type=int, default=1,help='output channel of network')

    return parser.parse_args()


args = get_args()

def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float: 
    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def haussdorf(preds, target):

    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()
    res = numpy_haussdorf(n_pred, n_target)
    return res

def dice_coeff(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return loss


def hunxiao(preds, target):
    tp = np.sum((preds == 1) & (target == 1))
    fp = np.sum((preds == 1) & (target == 0))
    tn = np.sum((preds == 0) & (target == 0))
    fn = np.sum((preds == 0) & (target == 1))

    smooh = 1e-10
    DI = 2*tp / (fp + 2*tp + fn)
    JA = tp / (tp + fn + fp)
    sensitivity = tp / (tp + fn)
    Accuracy = (tp + tn) / (tn + tp + fp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp + smooh)
    G = math.sqrt(sensitivity * specificity)
    # DI = 2 * tp / (fp + 2 * tp + fn + smooh)
    # JA = tp / (tp + fn + fp + smooh)
    # sensitivity = tp / (tp + fn + smooh)
    # Accuracy = (tp + tn) / (tn + tp + fp + fn + smooh)
    # specificity = tn / (tn + fp + smooh)
    # precision = tp / (tp + fp + smooh)

    return DI, JA, sensitivity, Accuracy, specificity, precision, G


def hunxiao1(preds, target):
    n_pred = preds.ravel()
    n_target = target.astype('int64').ravel()
    tn, fp, fn, tp = confusion_matrix(n_pred, n_target).ravel()

    smooh = 1e-10
    sensitivity = tp / (tp + fn + smooh)
    specificity = tn / (tn + fp + smooh)
    Accuracy = (tp + tn) / (tn + tp + fp + fn + smooh)
    precision = tp / (tp + fp + smooh)

    return sensitivity, specificity, Accuracy, precision


def predict_img(net, full_img, device, scale_factor=0.5, out_threshold=0.5):  #
    net.eval()
    with torch.no_grad():
        # output, _ = net(full_img)
        # _, output = net(full_img)
        a1, a2 = net(full_img)
        output = 0.5 * (a1 + a2)
        if args.num_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),  

                transforms.ToTensor() 
            ]
        )
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        res = np.int64(full_mask > out_threshold) 
    return res


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8)) 


def image_to_image(image):
    return Image.fromarray(image.astype(np.uint8))


if __name__ == "__main__":
    start = time.perf_counter()
    # if not os.path.exists('output/ISIC2018-1816-259-519'):
    #     os.makedirs('output/ISIC2018-1816-259-519')

    test_dataset = isic_loader(path_Data='input/ISIC2018-2594-100/', train=False, Test=False)

    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(len(test_dataset))
    net = CUM-Net(in_chns=3, num_classes=1)   

    logging.info("Loading model {}".format(args.model))

    logs = pds.DataFrame(index=[], columns=['iter_num', 'dice', 'hd_d'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load('model/' + args.model, map_location=device))  

    logging.info("Model loaded !")

    dc = []
    ja = []
    sensitivity = []
    Accuracy = []
    specificity = []
    hd = []
    Pre = []
    Gmean = []
    for i_batch, sampled_batch in enumerate(test_loader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch = volume_batch.to(device, dtype=torch.float32)
        mask_type = torch.float32 if args.num_classes == 1 else torch.long
        label_batch = label_batch.to(device, dtype=mask_type)

        pd = predict_img(net=net,
                           full_img=volume_batch,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        label_batch = label_batch.squeeze(0)  # torch.Size([1, 224, 224])
        label_batch = label_batch.squeeze(0)
        label_batch1 = label_batch.cpu().numpy()
        pdss = torch.from_numpy(pd).to(device)
        DI, JA, sen, Acc, spe, pre, G = hunxiao(pd, label_batch1)
        d = dice_coeff(pdss, label_batch).item()

        hd_d = haussdorf(pdss, label_batch)


        tmp = pds.Series([i_batch, d, hd_d], index=['iter_num', 'dice', 'hd_d'])
        logs = logs.append(tmp, ignore_index=True)  
        logs.to_csv('output/.csv', index=False)

        # result = mask_to_image(pd) 
        # if not os.path.exists('output/'):
        #     os.makedirs('output/')
        # save_dir = 'output/'
        # image_path = os.path.join(save_dir, f"prediction_{i_batch}.png")
        # result.save(image_path)

        dc.append(d)
        hd.append(hd_d)
        ja.append(JA)
        sensitivity.append(sen)
        Accuracy.append(Acc)
        specificity.append(spe)
        Pre.append(pre)
        Gmean.append(G)

    print('DICE: %.4f' % np.mean(dc))
    print('HD: %.4f' % np.mean(hd))
    print('JA: %.4f' % np.mean(ja))
    print('sensitivity: %.4f' % np.mean(sensitivity))
    print('Accuracy: %.4f' % np.mean(Accuracy))
    print('specificity: %.4f' % np.mean(specificity))
    print('precision: %.4f' % np.mean(Pre))
    print('G-mean: %.4f' % np.mean(Gmean))


    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
