from __future__ import print_function
import argparse
import math
import os
import torch
import sys
import time
import numpy as np
from tqdm import tqdm
import pprint
from dataset import CANDataset
from networks.rec_cnn import RecCNN
from networks.lstm_cnn import LSTMCNN
from networks.vision_trans import CEViT
from util import AverageMeter, save_model
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

import torch.nn.functional as F


# === ARGUMENT PARSER === #
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--trained_model_path', type=str, required=True, help='path to pre-trained model checkpoint')
    parser.add_argument('--data_folder', type=str, required=True, help='path to dataset')
    parser.add_argument('--n_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--save_path', type=str, default='2011_chevrolet_impala', help='path to save model')
    parser.add_argument('--version', type=str, default='v2', help='version')
    parser.add_argument('--model', type=str, default='rec_cnn', choices=['rec_cnn', 'lstm_cnn', 'vit'], help='model')
    # mixup
    parser.add_argument('--lamda', type=float, default=0.5, 
                        help='universum lambda')
    parser.add_argument('--mix', type=str, default='mixup', 
                        choices=['mixup', 'cutmix'], 
                        help='use mixup or cutmix')
    parser.add_argument('--size', type=int, default=32, 
                        help='parameter for RandomResizedCrop')
    opt = parser.parse_args()
    opt.save_folder = './save/CAN-ML_models/Transferred/Few_shot/' + opt.save_path + '/' + opt.version
    return opt


# === DATA LOADER === #
def set_loader(opt):
    train_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=True)
    test_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader


# === MODEL AND CRITERION === #
def set_model(opt):
    if opt.model == 'rec_cnn':
        model = RecCNN(num_classes=opt.n_classes)
    elif opt.model == 'lstm_cnn':
        model = LSTMCNN(num_classes=opt.n_classes)
    elif opt.model == 'vit':
        model = CEViT(emb_size=256, n_classes=opt.n_classes)    
    # checkpoint_path = os.path.join(opt.trained_model_path, 'ckpt_epoch_95.pth')
    checkpoint_path = os.path.join(opt.trained_model_path, 'ckpt_epoch_100.pth')
    checkpoint = torch.load(checkpoint_path, map_location=opt.device, weights_only=False)
    model.load_state_dict(checkpoint['model'])

    # Fine-tune all layers
    for param in model.parameters():
        param.requires_grad = True

    # criterion = UniConLoss(temperature=0.07)
    criterion = torch.nn.CrossEntropyLoss()

    return model.to(opt.device), criterion.to(opt.device)

# === OPTIMIZER === #
def set_optimizer(opt, model, is_classifier=False):
    if is_classifier:
        return torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        return torch.optim.SGD(model.parameters(), lr=opt.learning_rate * 0.1, momentum=opt.momentum, weight_decay=opt.weight_decay)


# === PROTOTYPES CALCULATE === #
def compute_prototypes(features, labels, n_classes):
    prototypes = []
    for c in range(n_classes):
        class_features = features[labels == c]
        if class_features.size(0) == 0:
            print(f"Class {c} has no samples. Setting prototype to mean of all other classes.")
            prototype = features.mean(dim=0)
        else:
            prototype = class_features.mean(dim=0)
            if torch.isnan(prototype).any() or torch.isinf(prototype).any():
                print(f"Prototype for class {c} contains NaN or Inf. Setting to mean of all other classes.")
                prototype = features.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)

def create_few_shot_batch(dataset, n_way, k_shot):
    few_shot_batch = []

    for c in range(n_way):
        if c not in dataset.class_indices or len(dataset.class_indices[c]) < k_shot:
            print(f"Class {c} does not have enough samples. Adding oversampling...")
            existing_samples = dataset.class_indices.get(c, [])
            while len(existing_samples) < k_shot:
                existing_samples += existing_samples[:k_shot - len(existing_samples)]
            selected_indices = torch.randperm(len(existing_samples))[:k_shot]
        else:
            selected_indices = torch.randperm(len(dataset.class_indices[c]))[:k_shot]

        for idx in selected_indices:
            data, label = dataset[dataset.class_indices[c][idx]]
            few_shot_batch.append((data, label))

    if len(few_shot_batch) == 0:
        print("Warning: Few-shot batch is empty after sampling.")

    return few_shot_batch


# === EPISODIC TRAINING === #
def episodic_training(train_loader, model, optimizer, opt, n_classes, n_way, k_shot):
    model.train()
    losses = AverageMeter()
    
    batch_time = AverageMeter()  # Time taken for each episode
    data_time = AverageMeter()   # Time taken for data processing
    
    end = time.time()  # Start timing

    for episode in range(len(train_loader)):
        data_time.update(time.time() - end) 
        
        # Create Few-Shot batch
        few_shot_batch = create_few_shot_batch(train_loader.dataset, n_way, k_shot)
        
        if len(few_shot_batch) == 0:
            print("Few-shot batch is empty. Skipping episode.")
            continue

        # Normalize and check the batch
        corrected_batch = []
        for i, (data, label) in enumerate(few_shot_batch):
            if isinstance(data, (list, tuple)):
                image1, image2 = data
                corrected_batch.append((image1, label))
                corrected_batch.append((image2, label))
            elif isinstance(data, torch.Tensor):
                corrected_batch.append((data, label))
            else:
                raise ValueError(f"Unexpected type for data: {type(data)}")

        # Separate images and labels
        images = torch.stack([item[0] for item in corrected_batch])
        labels = torch.tensor([item[1] for item in corrected_batch])

        # Move data to device
        images = images.to(opt.device)
        labels = labels.to(opt.device)

        # Extract features
        features = model(images)
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("Features contain NaN or Inf! Skipping this episode.")
            continue

        # Compute prototypes
        prototypes = compute_prototypes(features, labels, n_classes)
        if torch.isnan(prototypes).any() or torch.isinf(prototypes).any():
            print("Prototypes contain NaN or Inf! Skipping this episode.")
            continue
        
        # Compute distances
        distances = torch.cdist(features, prototypes)
        distances = torch.clamp(distances, min=1e-6, max=1e6)

        # Compute loss
        loss = F.cross_entropy(-distances, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss contains NaN or Inf! Skipping this batch.")
            continue

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss and time
        losses.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Log episode information
        if (episode + 1) % opt.print_freq == 0:
            print(
                f"Episode: [{episode + 1}/{len(train_loader)}]\t"
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})"
            )

    print(f"Episodic Training Loss: {losses.avg:.4f}")

# === TRAIN FUNCTIONS === #
def train_encoder(train_loader, model, criterion, optimizer, epoch, opt):
    """One epoch training with Cross-Entropy loss"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # Move data to device (GPU/CPU)
        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        # Forward pass
        outputs = model(images)  # Direct classification from the model

        # Calculate Cross-Entropy Loss
        loss = criterion(outputs, labels)

        # Update metric
        losses.update(loss.item(), bsz)

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure time and print information
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % opt.print_freq == 0:
            log_message = (
                'Train encoder: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.5f} ({data_time.avg:.5f})\t'
                'loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses
                )
            )
            print(log_message)
            sys.stdout.flush()
    print(f"Encoder Loss: {losses.avg:.4f}")

def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu().numpy().squeeze(0)
    return pred

def validate(val_loader, model, criterion, opt):
    """Validation"""
    print("Classifier...")
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    total_pred = np.array([], dtype=int)
    total_label = np.array([], dtype=int) 

    with torch.no_grad():
        end = time.time()
        for images, labels in tqdm(val_loader):
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)


            # Forward pass
            outputs = model(images)
            bsz = labels.size(0)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), bsz)

            # Update metric
            pred = get_predict(outputs)

            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)

            # Measure elapsed time
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


def adjust_learning_rate(args, optimizer, epoch, class_str=''):
    dict_args = vars(args)
    lr = dict_args['learning_rate'+class_str]
    eta_min = lr * (dict_args['lr_decay_rate'+class_str] ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# === MAIN FUNCTION === #
def main():
    opt = parse_option()
    print(opt)

    seed = 1
    torch.manual_seed(seed)

    # Load data
    train_loader, test_loader = set_loader(opt)

    # Initialize model, criterion, and optimizers
    model, criterion = set_model(opt)
    optimizer_encoder = set_optimizer(opt, model)       # Optimizer for encoder

    for epoch in range(1, opt.epochs + 1):
        torch.cuda.empty_cache()
        print(f"Epoch {epoch}/{opt.epochs}")
        print('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        print('=====================')
        
        adjust_learning_rate(opt, optimizer_encoder, epoch)

        print('Fine tune Encoder...')
        train_encoder(train_loader, model, criterion, optimizer_encoder, epoch, opt)
        
        print('Episodic Training...')
        episodic_training(train_loader, model, optimizer_encoder, opt, opt.n_classes, n_way=10, k_shot=15)

        print('Validation...')
        val_loss, val_acc = validate(test_loader, model, criterion, opt)

        # # Save model every 10 epochs
        # if epoch % 10 == 0:
        save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
        save_model(model, optimizer_encoder, opt, epoch, save_file)


if __name__ == '__main__':
    main()
