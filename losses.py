from __future__ import print_function
import torch

"""
Author: Aiyang Han (aiyangh@nuaa.edu.cn)
Date: May 24th, 2022
"""

import torch.nn as nn
import torch.nn.functional as F


class UniConLoss_Standard(nn.Module):
    """Universum-inspired Supervised Contrastive Learning: https://arxiv.org/abs/2204.10695"""

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(UniConLoss_Standard, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, universum, labels):
        """
        We include universum data into the calculation of InfoNCE.
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            universum: universum data of shape [bsz*n_views, ...]
        Returns:
            A loss scalar.
        """
        # Get device from `features`
        device = features.device

        # Check and synchronize device for tensors
        labels = labels.to(device)
        universum = universum.to(device)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # Synchronize device for `mask`
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # shape of [bsz*n_views, feature_dimension]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # only show one view, shape of [bsz, feature_dimension]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # show all the views, shape of [bsz*n_views, feature_dimension]
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, universum.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # find the biggest
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # make the size suited for similarity matrix

        # mask-out self-contrast cases, make value on the diagonal False
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class UniConLoss(nn.Module):
    def __init__(self, temperature=0.1, contrast_mode='all', base_temperature=0.1, 
                 gamma=2.0, alpha=0.25):
        super(UniConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, features, universum, labels):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, universum.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute probability p_ij
        prob = torch.exp(log_prob)

        # compute focal weight
        focal_weight = self.alpha * (1 - prob) ** self.gamma

        # compute class weights dynamically based on batch
        class_counts = torch.bincount(labels.view(-1))
        class_weights = torch.zeros_like(class_counts, dtype=torch.float, device=device)
        for c in range(len(class_counts)):
            if class_counts[c] > 0:
                class_weights[c] = batch_size / (len(class_counts) * class_counts[c])
        # Apply class weights to labels
        weights = class_weights[labels.view(-1)]

        if anchor_count > 1:
            weights = weights.repeat(anchor_count)

        # compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = ((mask * focal_weight) * log_prob).sum(1) / mask.sum(1)

        # Apply dynamic weights
        mean_log_prob_pos = mean_log_prob_pos * weights

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SupConLoss(nn.Module):
    """This part is from the pytorch implementation of SupCon.
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', epsilon=1e-6):
        """
        Args:
            alpha (float or Tensor): Weight for each class. If a number, apply evenly to all classes.
                                     If a Tensor, the size must match the number of classes.
            gamma (float): Weight adjustment parameter for easy and hard samples.
            reduction (str): Loss reduction mode, supports 'mean', 'sum', and 'none'.
            epsilon (float): Avoid log(0) or division by 0.
        """
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, (float, torch.Tensor)):
            raise TypeError("alpha must be float or Tensor")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # Compute cross-entropy loss (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # p_t: probability of the true class

        # Handle alpha (class weights)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Only take weights for classes appearing in the batch
                valid_classes = torch.unique(targets)
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha  # Single alpha value
        else:
            alpha_t = 1.0  # No weighting

        # Compute Focal Loss
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss

        # Avoid NaN by adding epsilon
        focal_loss = focal_loss.clamp(min=self.epsilon)

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, device='cpu'):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]  # Select prototypes for each class
        loss = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss
