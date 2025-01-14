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
from torchvision import transforms
import torch.nn.functional as F

# from dataset import CANDatasetEnet as CANDataset
from dataset import CANDataset
from losses import UniConLoss ,FocalLoss, CenterLoss
from networks.resnet_big import ConResNet, LinearClassifier
from util import AverageMeter, save_model, get_universum, TwoCropTransform, AddGaussianNoise
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from torch.utils.tensorboard import SummaryWriter


# === ARGUMENT PARSER === #
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
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
    parser.add_argument('--device', type=str, default='cuda:1', help='device to use')
    parser.add_argument('--save_path', type=str, default='2011_chevrolet_impala', help='path to save model')
    parser.add_argument('--version', type=str, default='v2', help='version')

    # mixup
    parser.add_argument('--lamda', type=float, default=0.5, 
                        help='universum lambda')
    parser.add_argument('--mix', type=str, default='mixup', 
                        choices=['mixup', 'cutmix'], 
                        help='use mixup or cutmix')
    parser.add_argument('--size', type=int, default=32, 
                        help='parameter for RandomResizedCrop')
    opt = parser.parse_args()
    opt.save_folder = './save/CAN-ML_models/Transferred/Unicon_Resnet/' + opt.save_path + '/' + opt.version
    return opt


# === DATA LOADER === #
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

    train_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=True, transform=TwoCropTransform(train_transform))
    test_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader


# === MODEL AND CRITERION === #
def set_model(opt):
    model = ConResNet(name='resnet50')
    classifier = LinearClassifier(num_classes=opt.n_classes)

    checkpoint_path = os.path.join(opt.trained_model_path, 'ckpt_epoch_37.pth')
    checkpoint = torch.load(checkpoint_path, map_location=opt.device, weights_only=False)
    # model.load_state_dict(checkpoint['model'])
         # Xử lý load state_dict với strict=False
    try:
        model.load_state_dict(checkpoint['model'], strict=False)
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
        model_state_dict = model.state_dict()
        pretrained_state_dict = checkpoint['model']
        for key in list(pretrained_state_dict.keys()):
            if key not in model_state_dict or model_state_dict[key].shape != pretrained_state_dict[key].shape:
                del pretrained_state_dict[key]
        model.load_state_dict(pretrained_state_dict, strict=False)
    # Fine-tune all layers
    for param in model.parameters():
        param.requires_grad = True

    criterion = UniConLoss(temperature=0.07)
    criterion_classifier = torch.nn.CrossEntropyLoss()

    return model.to(opt.device), classifier.to(opt.device), criterion.to(opt.device), criterion_classifier.to(opt.device)

# === OPTIMIZER === #
def set_optimizer(opt, model, is_classifier=False):
    if is_classifier:
        return torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    else:
        return torch.optim.SGD(model.parameters(), lr=opt.learning_rate * 0.1, momentum=opt.momentum, weight_decay=opt.weight_decay)

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
    
    batch_time = AverageMeter()  # Thời gian xử lý mỗi episode
    data_time = AverageMeter()   # Thời gian xử lý dữ liệu
    
    end = time.time()  # Bắt đầu đo thời gian

    for episode in range(len(train_loader)):
        data_time.update(time.time() - end) 
        
        # Tạo batch Few-Shot
        few_shot_batch = create_few_shot_batch(train_loader.dataset, n_way, k_shot)
        
        if len(few_shot_batch) == 0:
            print("Few-shot batch is empty. Skipping episode.")
            continue

        # Chuẩn hóa và kiểm tra batch
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

        # Tách images và labels
        images = torch.stack([item[0] for item in corrected_batch])
        labels = torch.tensor([item[1] for item in corrected_batch])

        # Chuyển dữ liệu sang thiết bị
        images = images.to(opt.device)
        labels = labels.to(opt.device)

        # Extract features
        features = model.encoder(images)
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("Features contain NaN or Inf! Skipping this episode.")
            continue

        # Compute prototypes
        prototypes = compute_prototypes(features, labels, n_classes)
        if torch.isnan(prototypes).any() or torch.isinf(prototypes).any():
            print("Prototypes contain NaN or Inf! Skipping this episode.")
            continue

        center_loss = CenterLoss(num_classes=n_classes, feature_dim=features.size(1), device=opt.device).to(opt.device)
        center_loss_weight = 0.01  # The lambda coefficient to adjust the impact level

        # Compute distances
        distances = torch.cdist(features, prototypes)
        distances = torch.clamp(distances, min=1e-6, max=1e6)
        
        # Compute loss
        loss_cross_entropy = F.cross_entropy(-distances, labels)
        loss_center = center_loss(features, labels)
        loss = loss_cross_entropy + center_loss_weight * loss_center
        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss contains NaN or Inf! Skipping this batch.")
            continue

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Cập nhật loss và thời gian
        losses.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Log thông tin mỗi episode
        if (episode + 1) % opt.print_freq == 0:
            print(
                f"Episode: [{episode + 1}/{len(train_loader)}]\t"
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})"
            )
            sys.stdout.flush()
    print(f"Episodic Training Loss: {losses.avg:.4f}")

# === TRAIN FUNCTIONS === #
def train_encoder(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        image1, image2 = images[0], images[1]
        images = torch.cat([image1, image2], dim=0)

        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]

        universum = get_universum(images, labels, opt).to(opt.device)
        uni_features = model(universum)

        # Forward pass
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # Compute UniCon loss
        loss = criterion(features, uni_features, labels)
        losses.update(loss.item(), bsz)

        # Backward pass và optimization   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % opt.print_freq == 0:
            log_message = (
                'Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\n'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses
                )
            )
            print(log_message)
            sys.stdout.flush()
    print(f"Encoder Loss: {losses.avg:.4f}")
def train_classifier(train_loader, model, classifier, criterion, optimizer, opt):
    model.eval()
    classifier.train()
    losses = AverageMeter()

    for images, labels in train_loader:
        images = images[0].to(opt.device)
        labels = labels.to(opt.device)

        with torch.no_grad():
            features = model.encoder(images)

        outputs = classifier(features.detach())
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), labels.size(0))

    print(f"Classifier Loss: {losses.avg:.4f}")

def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu().numpy().squeeze(0)
    return pred

def validate(val_loader, model, classifier, criterion, opt):
    model.eval()
    classifier.eval()
    
    losses = AverageMeter()
    total_pred = np.array([], dtype=int)
    total_label = np.array([], dtype=int) 
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader): 
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)
            features = model.encoder(images)
            outputs = classifier(features)

            bsz = labels.size(0)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), bsz)
            
            pred = get_predict(outputs)

            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)
    
    acc = accuracy_score(total_label, total_pred)
    acc = acc * 100
    f1 = f1_score(total_label, total_pred, average='weighted')
    precision = precision_score(total_label, total_pred, average='weighted', zero_division=0)
    recall = recall_score(total_label, total_pred, average='weighted')
    conf_matrix = confusion_matrix(total_label, total_pred)
    
    return losses.avg, acc, f1, precision, recall, conf_matrix


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
    model, classifier, criterion, criterion_classifier = set_model(opt)
    optimizer_encoder = set_optimizer(opt, model)       # Optimizer for encoder
    optimizer_classifier = set_optimizer(opt, classifier, is_classifier=True)  # Optimizer for classifier

    for epoch in range(1, opt.epochs + 1):
        torch.cuda.empty_cache()
        print(f"Epoch {epoch}/{opt.epochs}")
        print('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        print('=====================')
        adjust_learning_rate(opt, optimizer_encoder, epoch)
        # adjust_freeze_layers(epoch, model, optimizer_encoder, freeze_epoch=15)
        print('Fine-tune Encoder...')
        train_encoder(train_loader, model, criterion, optimizer_encoder, epoch, opt)
        
        print('Episodic Training...')
        episodic_training(train_loader, model, optimizer_encoder, opt, opt.n_classes, n_way=10, k_shot=5)
        
        print('Fine-tune Classifier...')
        train_classifier(train_loader, model, classifier, criterion_classifier, optimizer_classifier, opt)

        # Step 4: Validation
        print('Validation...')
        loss, val_acc, val_f1, precision, recall, conf_matrix = validate(test_loader, model, classifier, criterion_classifier, opt)
        print(f'Validation Loss: {loss:.4f}, Accuracy: {val_acc:.4f}')
        print(f'F1 Score: {val_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)

        # # Save model every 10 epochs
        # if epoch % 10 == 0:
        save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
        save_model(model, optimizer_encoder, opt, epoch, save_file)

        save_file = os.path.join(opt.save_folder, f'ckpt_class_epoch_{epoch}.pth')
        save_model(classifier, optimizer_classifier, opt, epoch, save_file)


if __name__ == '__main__':
    main()
