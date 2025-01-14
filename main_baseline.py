from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as F
import pprint
from dataset import CANDatasetEnet as CANDataset
from networks.rec_cnn import RecCNN
from networks.lstm_cnn import LSTMCNN
from datetime import datetime
from util import AverageMeter
from util import warmup_learning_rate
from util import save_model ,load_checkpoint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--resume', type=str, default=None, 
                        help='path to the checkpoint to resume from')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # optimization classifier
    parser.add_argument('--epoch_start_classifier', type=int, default=50)
    parser.add_argument('--learning_rate_classifier', type=float, default=0.01,
                        help='learning rate classifier')
    parser.add_argument('--lr_decay_epochs_classifier', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate_classifier', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay_classifier', type=float, default=0,
                        help='weight decay_classifier')
    parser.add_argument('--momentum_classifier', type=float, default=0.9,
                        help='momentum_classifier')
    # model dataset
    parser.add_argument('--model', type=str, default='rec_cnn', choices=['rec_cnn', 'lstm_cnn'], help='model')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--dataset', type=str, default='CAN',
                        choices=['CAN-ML', 'CAN', 'ROAD'], 
                        help='dataset')
    parser.add_argument('--mean', type=str, 
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, 
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, 
                        help='path to custom dataset')
    parser.add_argument('--n_classes', type=int, default=5, 
                        help='number of class')

    # method
    parser.add_argument('--method', type=str, default='BaseLine',
                        help='choose method')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    opt = parser.parse_args()

    opt.device = torch.device('cuda:0')
    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './data/Car-Hacking/TFRecord_w32_s32/2/'
    opt.model_path = './save/{}_models/{}'.format(opt.dataset, opt.method)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    current_time = datetime.now().strftime("%D_%H%M%S").replace('/', '')
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'. \
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.trial)

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
    # dir to save log file
    opt.log_file = f'{opt.model_path}/{opt.model_name}/log'
    opt.tb_folder = f'{opt.model_path}/{opt.model_name}/runs'
    return opt

def set_loader(opt):
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)
    # normalize = transforms.Normalize(mean=mean, std=std)

    # transform = transforms.Compose([normalize])

    train_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=True)
    test_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=False, include_data=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader


def set_model(opt):
    # torch.cuda.empty_cache()
    if opt.model == 'rec_cnn':
        model = RecCNN(num_classes=opt.n_classes)
    elif opt.model == 'lstm_cnn':
        model = LSTMCNN(num_classes=opt.n_classes)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_validate = torch.nn.CrossEntropyLoss()

    model = model.to(opt.device)
    criterion = criterion.to(opt.device)
    criterion_validate = criterion_validate.to(opt.device)

    # print('Model device: ', next(model.parameters()).device)
    return model, criterion, criterion_validate

optimize_dict = {
    'SGD' : optim.SGD,
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW
}

def set_optimizer(opt, model, class_str='', optim_choice='SGD'):
    dict_opt = vars(opt)
    optimizer = optimize_dict[optim_choice]

    # Handle both Adam and AdamW cases
    if optim_choice in ['Adam', 'AdamW']:
        optimizer = optimizer(
            model.parameters(),
            lr=dict_opt['learning_rate' + class_str],
            weight_decay=dict_opt['weight_decay' + class_str]
        )
    else:
        # For optimizers that require momentum like SGD and RMSprop
        optimizer = optimizer(
            model.parameters(),
            lr=dict_opt['learning_rate' + class_str],
            momentum=dict_opt['momentum' + class_str],
            weight_decay=dict_opt['weight_decay' + class_str]
        )

    return optimizer


def train(train_loader, model, criterion, optimizer, epoch, opt, step):
    """One epoch training with Cross-Entropy loss"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        step += 1
        data_time.update(time.time() - end)
        
        # Đưa dữ liệu vào thiết bị (GPU/CPU)
        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        # Điều chỉnh learning rate (nếu cần)
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # Tiến hành truyền dữ liệu qua mô hình
        outputs = model(images)  # Phân loại trực tiếp từ mô hình

        # Tính toán Cross-Entropy Loss
        loss = criterion(outputs, labels)

        # Cập nhật metric
        losses.update(loss.item(), bsz)

        # Thực hiện bước tối ưu hóa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Đo thời gian và in thông tin
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % opt.print_freq == 0:
            log_message = (
                'Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.5f} ({data_time.avg:.5f})\t'
                'loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses
                )
            )
            print(log_message)
            sys.stdout.flush()

    return step, losses.avg

def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu().numpy().squeeze(0)
    return pred

def validate(val_loader, model, criterion, opt):
    model.eval()
    
    losses = AverageMeter()
    total_pred = np.array([], dtype=int)
    total_label = np.array([], dtype=int) 
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader): 
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)

            outputs = model(images)
            bsz = labels.size(0)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), bsz)
            
            pred = get_predict(outputs)

            # total_pred = np.concatenate((total_pred, pred), axis=0)
            # total_label = np.concatenate((total_label, labels), axis=0)
            # Nếu `pred` đã là NumPy array, không cần gọi `.cpu()` nữa
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            
            # Chuyển đổi `labels` sang NumPy nếu cần
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            # Cập nhật các giá trị tổng
            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)
    
    f1 = f1_score(total_label, total_pred, average='weighted')
    precision = precision_score(total_label, total_pred, average='weighted', zero_division=0)
    recall = recall_score(total_label, total_pred, average='weighted')
    conf_matrix = confusion_matrix(total_label, total_pred)
    accuracy = accuracy_score(total_label, total_pred)
    accuracy = accuracy * 100
    return losses.avg, f1, precision, recall, conf_matrix, accuracy


def adjust_learning_rate(args, optimizer, epoch, class_str=''):
    dict_args = vars(args)
    lr = dict_args['learning_rate'+class_str]
    if args.cosine:
        eta_min = lr * (dict_args['lr_decay_rate'+class_str] ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        lr_decay_epochs_arr = list(map(int, dict_args['lr_decay_epochs'+class_str].split(',')))
        lr_decay_epochs_arr = np.asarray(lr_decay_epochs_arr)
        steps = np.sum(epoch > lr_decay_epochs_arr)
        if steps > 0:
            lr = lr * (dict_args['lr_decay_rate'+class_str] ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    opt = parse_option()
    print(opt)

    seed = 1
    torch.manual_seed(seed)

    # build data loader
    train_loader, test_loader = set_loader(opt)
    model, criterion, criterion_validate = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model, optim_choice=opt.optimizer)
    
    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    start_epoch = 1  # Default start from epoch 1
    step = 0

    # Loading from a checkpoint
    if opt.resume:
        new_epoch = opt.epochs
        new_save_freq = opt.save_freq
        checkpoint_path = opt.model_path + '/' + opt.model_name + '/' + opt.resume
        model, optimizer, start_epoch, opt = load_checkpoint(checkpoint_path, model, optimizer)

        if new_epoch != opt.epochs:
            opt.epochs = new_epoch
        if new_save_freq != opt.save_freq:
            opt.save_freq = new_save_freq
        print(f"Resuming training from epoch {start_epoch} to epoch {opt.epochs}...")

    log_writer = open(opt.log_file, 'w')
    # training routine
    pp = pprint.PrettyPrinter(indent=4)

    for epoch in range(start_epoch, opt.epochs + 1):
        print('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        adjust_learning_rate(opt, optimizer, epoch)

        new_step, train_loss = train(train_loader, model, criterion, optimizer, epoch, opt, step)

        print(f'Epoch {epoch}, Unicon Loss {train_loss:.4f}')
        log_writer.write(f'Epoch: {epoch}, Unicon Loss: {train_loss:.4f}\n')
        logger.add_scalar('train_loss/val', train_loss, step)
        
        loss, val_f1, precision, recall, conf_matrix, accuracy = validate(test_loader, model, criterion_validate, opt)
        logger.add_scalar('loss_ce/val', loss, step)
        logger.add_scalar('acc/val', accuracy, step)
        logger.add_scalar('val_f1/val', val_f1, step)

        log_writer.write(f'Accuracy: {accuracy:.4f}\n')
        log_writer.write(f'F1 Score: {val_f1:.4f}\n')
        log_writer.write(f'Precision: {precision:.4f}\n')
        log_writer.write(f'Recall: {recall:.4f}\n')
        log_writer.write(f'Confusion Matrix:\n{pp.pformat(conf_matrix)}\n')
            
        # Print results
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {val_f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)

        step = new_step
        # Save checkpoint 
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)
    
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    
if __name__ == '__main__':
    main()