from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from dataset import CANDatasetEnet as CANDataset
from tqdm import tqdm 

from networks.resnet_big import CEResNet
from util import AverageMeter, TwoCropTransform, AddGaussianNoise
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from torch.utils.tensorboard import SummaryWriter

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='CAN-ML',
                        choices=['CAN', 'ROAD', 'CAN-ML', 'CAN-TT'],
                        help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, 
                    help='path to custom dataset')
    parser.add_argument('--n_classes', type=int, default=5, 
                    help='number of class')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()
    opt.device = torch.device('cuda:1')
    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './data/Car-Hacking/TFRecord_w32_s32/2/'
    opt.model_path = './save/{}_models/CE'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'CE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    opt.tb_folder = f'{opt.model_path}/{opt.model_name}/runs'
    return opt


def set_loader(opt):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.5),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform = transforms.Compose([normalize])
    
    train_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=True,
                transform=TwoCropTransform(transform=train_transform))
    test_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=False, include_data=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader


def set_model(opt):
    model = CEResNet(name=opt.model, num_classes=opt.n_classes)
    criterion = torch.nn.CrossEntropyLoss()

    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    criterion = criterion.to(opt.device)
    # cudnn.benchmark = True
    print('Model device: ', next(model.parameters()).device)
    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, step):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        step += 1
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        # images = images.cuda(non_blocking=True)
        # labels = labels.cuda(non_blocking=True)
        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        # with torch.no_grad():
        output = model(images)
        loss = criterion(output, labels)
        # update metric
        losses.update(loss.item(), bsz)
        # acc = accuracy(output, labels)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  .format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return step, losses.avg

def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu().numpy().squeeze(0)
    return pred

def validate(val_loader, model, criterion, opt):
    """validation"""
    print("Classifier...")
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    total_pred = np.array([], dtype=int)
    total_label = np.array([], dtype=int) 

    with torch.no_grad():
        end = time.time()
        for images, labels in tqdm(val_loader):
            images = images.float().to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)
            # images = images.float().cuda()
            # labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # update metric
            losses.update(loss.item(), bsz)
            pred = get_predict(outputs)

            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    acc = accuracy_score(total_label, total_pred)
    acc_percent = acc * 100
    f1 = f1_score(total_label, total_pred, average='weighted')
    precision = precision_score(total_label, total_pred, average='weighted', zero_division=0)
    recall = recall_score(total_label, total_pred, average='weighted')
    conf_matrix = confusion_matrix(total_label, total_pred)
    # Print results
    print('Val Classifier: Loss: {:.4f}, Acc: {}'.format(losses.avg, acc_percent))
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)

    return losses.avg, acc_percent


def main():
    opt = parse_option()
    print(opt)

    seed = 1
    torch.manual_seed(seed)
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)
    start_epoch = 1
    step = 0

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):

        print('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        new_step, train_loss = train(train_loader, model, criterion, optimizer, epoch, opt, step)
        print(f'Epoch {epoch}, Train Loss {train_loss:.4f}')

        # evaluation
        val_loss, val_acc = validate(val_loader, model, criterion, opt)
        logger.add_scalar('loss_ce/val', val_loss, step)
        logger.add_scalar('acc_ce/val', val_acc, step)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
    step = new_step
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
