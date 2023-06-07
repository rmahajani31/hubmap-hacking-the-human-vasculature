import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

from torchmetrics import Metric
class IoUScore(Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("intersection_back", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("union_back", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("intersection_fore", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("union_fore", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_images", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = (preds > self.threshold).int()
        intersection_back = torch.logical_and(preds[:,0,:,:], target==0).sum()
        union_back = torch.logical_or(preds[:,0,:,:], target==0).sum()
        intersection_fore = torch.logical_and(preds[:,1,:,:], target==1).sum()
        union_fore = torch.logical_or(preds[:,1,:,:], target==1).sum()

        self.intersection_back += intersection_back
        self.union_back += union_back
        self.intersection_fore += intersection_fore
        self.union_fore += union_fore

    def compute(self):
        iou_back = (self.intersection_back.float() / self.union_back.float())
        iou_fore = (self.intersection_fore.float() / self.union_fore.float())
        self.intersection_back = 0
        self.union_back = 0
        self.intersection_fore = 0
        self.union_fore = 0
        return iou_back,iou_fore

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    metrics = [
    IoUScore(threshold=0.5).cuda(),
    ]
    ce_loss = CrossEntropyLoss(weight=torch.tensor([0.05,1], dtype=torch.float32).cuda())
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        epoch_losses = []
        epoch_losses_ce = []
        epoch_start_time = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, weight=[0.05,1], softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            epoch_losses.append(loss.item())
            epoch_losses_ce.append(loss_ce.item())
            print(f'Shapes: {outputs.shape}, {label_batch.shape}')
            for metric in metrics:
                metric.update(torch.softmax(outputs, dim=1), label_batch)

            #logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item())) 

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        metric_values = [metric.compute() for metric in metrics]
        logging.info('epoch %d : loss : %f, loss_ce: %f, train_IoU_back: %f, train_IoU_fore: %f, epoch_time: %f' % (epoch_num, np.mean(epoch_losses), np.mean(epoch_losses_ce), metric_values[0][0], metric_values[0][1], float(time.time()-epoch_start_time)/60))
        save_interval = 5  # int(max_epoch/6)
        if epoch_num > 0 and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
