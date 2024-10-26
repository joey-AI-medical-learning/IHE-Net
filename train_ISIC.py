import logging
import os
import random
import pandas as pd
import shutil
import sys
from glob import glob
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss,BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from parameter_ISIC import get_parameters
from utils.load_ISIC import (isic_loader, TwoStreamBatchSampler)
from utils import ramps
from utils.losses import dice_loss
from val_ISIC import eval_modal

from network.IHE_Net(C_T) import CUT-Net
from network.IHE_Net(C_M) import CUM-Net

torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_parameters()


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = CUT-Net(3, num_classes).to(device)
        if ema:
            for param in model.parameters():
                param.detach_() 
        return model

    model = create_model() 
    ema_model = create_model(ema=True)  

    def worker_init_fn(worker_id): 
        random.seed(args.seed + worker_id)

    train_dataset = isic_loader(path_Data='input/ISIC2018-2594-100/', train=True) 
    val_dataset = isic_loader(path_Data='input/ISIC2018-2594-100/', train=False, Test=False) 

    val_len = len(val_dataset)
    total_slices = len(train_dataset)
    labeled_slice = args.labeled_num 

    print("Total silices is: {}, labeled slices is: {}, val_len is {}".format(
        total_slices, labeled_slice, val_len))
    labeled_idxs = list(range(0, labeled_slice)) 
    unlabeled_idxs = list(range(labeled_slice, total_slices)) 
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = BCELoss()
    logs = pd.DataFrame(index=[], columns=['iter_num', 'performance', 'mean_hd95'])
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch = volume_batch.to(device, dtype=torch.float32)
            mask_type = torch.float32 if num_classes == 1 else torch.long
            label_batch = label_batch.to(device, dtype=mask_type)

            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            output, output1 = model(volume_batch)
            output2 = output + output1
            output_soft = torch.sigmoid(output)
            output_soft1 = torch.sigmoid(output1)
            output_soft2 = torch.sigmoid(output2)
            with torch.no_grad():
                ema_output, ema_output1 = ema_model(ema_inputs)
                ema_output2 = ema_output + ema_output1
                ema_output_soft = torch.sigmoid(ema_output)
                ema_output_soft1 = torch.sigmoid(ema_output1)
                ema_output_soft2 = torch.sigmoid(ema_output2)

            loss_ce1 = 0.5 * (ce_loss(output_soft[:args.labeled_bs], label_batch[:args.labeled_bs]) + ce_loss(output_soft1[:args.labeled_bs], label_batch[:args.labeled_bs]) )
            loss_dice1 = 0.5 * (dice_loss(output_soft[:args.labeled_bs], label_batch[:args.labeled_bs]) + dice_loss(output_soft1[:args.labeled_bs], label_batch[:args.labeled_bs]))
            supervised_loss1 = 0.5 * (loss_dice1 + loss_ce1)

            consistency_weight = get_current_consistency_weight(iter_num//150)

            if iter_num < 1000: 
                consistency_loss0 = 0.0
                consistency_loss1 = 0.0
                consistency_loss2 = 0.0
            else:
                consistency_loss0 = torch.mean((output_soft[args.labeled_bs:] - ema_output_soft) ** 2)
                consistency_loss1 = torch.mean((output_soft1[args.labeled_bs:] - ema_output_soft1) ** 2)
                consistency_loss2 = torch.mean((output_soft2[args.labeled_bs:] - ema_output_soft2) ** 2)

            consistency_loss = consistency_loss0 + consistency_loss1 + consistency_loss2
            loss = supervised_loss1 + consistency_weight * consistency_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)


            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('loss/model_loss',
                              loss, iter_num)

            logging.info(
                'iteration %d : model_loss : %f ' % (iter_num, loss.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                performance, mean_hd95 = eval_modal(model, valloader, device, val_len, num_classes)
                # performance = eval_modal(model, valloader, device, val_len, num_classes)
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : model_mean_dice : %f model_mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model_iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed) 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "model/{}/{}_{}_labeled/{}".format( 
        'ISIC2018-2594-100', args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):  
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,  
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) 
    logging.info(str(args)) 
    train(args, snapshot_path) 
