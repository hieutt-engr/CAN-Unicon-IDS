from __future__ import print_function
import os
import math
import random
import json
import pickle
import codecs
import torch
import datetime
import numpy as np
import pandas as pd

import torch.optim as optim

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine and epoch <= 1000:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2  # args.epochs
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def rand_bbox(size, lam):
    '''Getting the random box in CutMix'''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_universum_standard(images, labels, opt):
    """Calculating Mixup-induced universum from a batch of images"""
    tmp = images.cpu()
    label = labels.cpu()
    bsz = tmp.shape[0]
    bs = len(label)
    class_images = [[] for i in range(max(label) + 1)]
    for i in label.unique():
        class_images[i] = np.where(label != i)[0]
    units = [tmp[random.choice(class_images[labels[i % bs]])] for i in range(bsz)]
    universum = torch.stack(units, dim=0).cuda()
    lamda = opt.lamda
    if not hasattr(opt, 'mix') or opt.mix == 'mixup':
        # Using Mixup
        universum = lamda * universum + (1 - lamda) * images
    else:
        # Using CutMix
        lam = 0
        while lam < 0.45 or lam > 0.55:
            # Since it is hard to control the value of lambda in CutMix,
            # we accept lambda in [0.45, 0.55].
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lamda)
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        universum[:, :, bbx1:bbx2, bby1:bby2] = images[:, :, bbx1:bbx2, bby1:bby2]
    return universum

def get_universum(images, labels, opt):
    """Calculating Mixup- or CutMix-induced universum from a batch of images
    Args:
        images: Tensor of input images (batch_size, C, H, W)
        labels: Tensor of input labels (batch_size)
        opt: Options containing settings like lambda and mixup type
        all_labels: (Optional) List of all labels in the dataset (used for extended universum)
        all_images: (Optional) List of all images in the dataset (used for extended universum)
    Returns:
        universum: Tensor of generated universum images
    """
    tmp = images.cpu()
    label = labels.cpu()
    bsz = tmp.shape[0]
    bs = len(label)

    # Prepare class-based image lists
    class_images = [[] for _ in range(max(label) + 1)]
    for i in label.unique():
        class_images[i] = np.where(label != i)[0]

    # Compute class counts and probabilities for balancing
    class_counts = [len(class_images[c]) for c in range(len(class_images))]
    epsilon = 1e-10
    class_counts = np.array(class_counts, dtype=float)
    weights = 1 / (class_counts + epsilon)  # Add epsilon to avoid division by zero

    # Normalize weights to probabilities
    if weights.sum() > 0:
        probabilities = weights / weights.sum()
    else:
        probabilities = np.zeros_like(weights)
    # Generate universum samples
    units = []
    for i in range(bsz):
        current_label = labels[i % bs].item()

        # Select other classes with available samples
        other_classes = [c for c in range(len(class_images)) if c != current_label and len(class_images[c]) > 0]

        if not other_classes:
            fallback_classes = [c for c in range(len(class_images)) if len(class_images[c]) > 0]
            if not fallback_classes:
                units.append(tmp[random.choice(range(bsz))])  # Fallback to the current batch
                continue
            selected_class = random.choice(fallback_classes)
        else:
            selected_class = random.choices(other_classes, weights=[probabilities[c] for c in other_classes])[0]

        # Select a random sample from the selected class
        selected_index = random.choice(class_images[selected_class])
        units.append(tmp[selected_index])

    universum = torch.stack(units, dim=0).to(opt.device)

    # Mixup or CutMix
    lamda = opt.lamda
    if not hasattr(opt, 'mix') or opt.mix == 'mixup':
        # Using Mixup
        universum = lamda * universum + (1 - lamda) * images
    elif opt.mix == 'cutmix':
        # Using CutMix
        lam = 0
        while lam < 0.45 or lam > 0.55:
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lamda)
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        universum[:, :, bbx1:bbx2, bby1:bby2] = images[:, :, bbx1:bbx2, bby1:bby2]
    else:
        print("Warning: No valid mixup method specified. Returning original universum.")

    return universum



def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, save_file)
    del state


def load_checkpoint(checkpoint_path, model, optimizer):
    print(f"==> Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])  # Load model weights
    optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer state
    start_epoch = checkpoint['epoch']  # Load saved epoch
    opt = checkpoint['opt']  # Load saved options if needed

    return model, optimizer, start_epoch, opt

# ---------- Other -----------

# functions for saving/opening objects
def jsonify(obj, out_file):
    """
    Inputs:
    - obj: the object to be jsonified
    - out_file: the file path where obj will be saved
    This function saves obj to the path out_file as a json file.
    """
    json.dump(obj, codecs.open(out_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def unjsonify(in_file):
    """
    Input:
    -in_file: the file path where the object you want to read in is stored
    Output:
    -obj: the object you want to read in
    """
    obj_text = codecs.open(in_file, 'r', encoding='utf-8').read()
    obj = json.loads(obj_text)
    return obj

def picklify(obj, filepath):
    """
    Inputs:
    - obj: the object to be pickled
    - filepath: the file path where obj will be saved
    This function pickles obj to the path filepath.
    """
    pickle_file = open(filepath, "wb")
    pickle.dump(obj, pickle_file)
    pickle_file.close()
    #print "picklify done"


def unpickle(filepath):
    """
    Input:
    -filepath: the file path where the pickled object you want to read in is stored
    Output:
    -obj: the object you want to read in
    """
    pickle_file = open(filepath, 'rb')
    obj = pickle.load(pickle_file)
    pickle_file.close()
    return obj

def curtime_str():
    """A string representation of the current time."""
    dt = datetime.datetime.now().time()
    return dt.strftime("%H:%M:%S")


def update_json_dict(key, value, out_file, overwrite = True):
    if not os.path.isfile(out_file):
        d = {}
    else:
        d = unjsonify(out_file)
        if key in d and not overwrite:
            print("fkey {key} already in {out_file}, skipping...")
            return
    d[key] = value
    jsonify(d, out_file)

    #jsonify(sorted(d.items(), key = lambda x: x[0]), out_file)


def make_can_df(log_filepath):
    """
    Puts candump data into a dataframe with columns 'time', 'aid', and 'data'
    """
    can_df = pd.read_fwf(
        log_filepath, delimiter = ' '+ '#' + '('+')',
        skiprows = 1,skipfooter=1,
        usecols = [0,2,3],
        dtype = {0:'float64', 1:str, 2: str},
        names = ['time','aid', 'data'] )

    can_df.aid = can_df.aid.apply(lambda x: int(x,16))
    can_df.data = can_df.data.apply(lambda x: x.zfill(16)) #pad with 0s on the left for data with dlc < 8
    can_df.time = can_df.time - can_df.time.min()
    
    return can_df[can_df.aid<=0x700] # out-of-scope aid


def add_time_diff_per_aid_col(df, order_by_time = False):
    """
    Sorts df by aid and time and takes time diff between each successive col and puts in col "time_diffs"
    Then removes first instance of each aids message
    Returns sorted df with new column
    """

    df.sort_values(['aid','time'], inplace=True)
    df['time_diffs'] = df['time'].diff()
    mask = df.aid == df.aid.shift(1) #get bool mask of to filter out first msg of each group
    df = df[mask]
    if order_by_time:
        df = df.sort_values('time').reset_index()
    return df


def get_injection_interval(df, injection_aid, injection_data_str, max_injection_t_delta=1):
    """
    Compute time intervals where attacks were injected based on aid and payload
    @param df: testing df to be analyzed (dataframe)
    @param injection_aid: aid that injects the attack (int)
    @param injection_data_str: payload of the attack (str)
    @param max_injection_t_delta: minimum separation between attacks (int)
    @output injection_intervals: list of intervals where the attacks were injected (list)
    """
    
    # Construct a regular expression to identify the payload
    injection_data_str = injection_data_str.replace("X", ".")

    attack_messages_df = df[(df.aid==injection_aid) & (df.data.str.contains(injection_data_str))] # get subset of attack messages
    #print(attack_messages_df)

    if len(attack_messages_df) == 0:
        print("message not found")
        return None

    # Assuming that attacks are injected with a diferrence more than i seconds
    inj_period_times = np.split(np.array(attack_messages_df.time),
        np.where(attack_messages_df.time.diff()>max_injection_t_delta)[0])

    # Pack the intervals
    injection_intervals = [(time_arr[0], time_arr[-1])
        for time_arr in inj_period_times if len(time_arr)>1]

    return injection_intervals


def add_actual_attack_col(df, intervals, aid, payload, attack_name):
    """
    Adds column to df to indicate which signals were part of attack
    """

    if aid != "XXX":
        if attack_name.startswith('correlated_signal'):
            df['label'] = df.time.apply(lambda x: sum(x >= intvl[0]  and x <= intvl[1] for intvl in intervals ) >= 1) & (df.aid == aid) & (df.data == payload)
        elif attack_name.startswith('max'):
            df['label'] = df.time.apply(lambda x: sum(x >= intvl[0]  and x <= intvl[1] for intvl in intervals ) >= 1) & (df.aid == aid) & df.data.str.contains(payload[10:12], regex=False)
        else:
            df['label'] = df.time.apply(lambda x: sum(x >= intvl[0]  and x <= intvl[1] for intvl in intervals ) >= 1) & (df.aid == aid) & df.data.str.contains(payload[4:6], regex=False)
    else:
        df['label'] = df.time.apply(lambda x: sum(x >= intvl[0]  and x <= intvl[1] for intvl in intervals ) >= 1) & (df.data == payload)
    return df
