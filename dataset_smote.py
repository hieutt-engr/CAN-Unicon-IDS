from imblearn.over_sampling import SMOTE
import numpy as np
import torch
import os
from torchvision import transforms
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import Dataset

class CANDataset(Dataset):
    def __init__(self, root_dir, window_size, is_train=True, include_data=False, transform=None, apply_smote=False):
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')
            
        self.include_data = include_data
        self.is_train = is_train
        self.transform = transform  # This will be TwoCropTransform
        self.window_size = window_size
        self.total_size = len(os.listdir(self.root_dir))
        self.apply_smote = apply_smote

        # Load all data if SMOTE is required
        if self.is_train and self.apply_smote:
            self.data, self.labels = self.load_data()
            self.apply_smote_to_data()
        else:
            self.data = None
            self.labels = None

    def load_data(self):
        """Load all data and labels from the TFRecord files."""
        data, labels = [], []
        for idx in range(self.total_size):
            filenames = '{}/{}.tfrec'.format(self.root_dir, idx)
            index_path = None
            description = {'id_seq': 'int', 'data_seq': 'int', 'timestamp': 'float', 'label': 'int'}
            dataset = TFRecordDataset(filenames, index_path, description)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
            batch = next(iter(dataloader))

            id_seq, data_seq, timestamp, label = batch['id_seq'], batch['data_seq'], batch['timestamp'], batch['label']
            id_seq = id_seq.to(torch.float)
            data_seq = data_seq.to(torch.float)
            timestamp = timestamp.to(torch.float)

            id_seq[id_seq == 0] = -1
            if id_seq.numel() == 1024 and data_seq.numel() == 1024:
                id_seq = id_seq.view(32, 32)
                data_seq = data_seq.view(32, 32)
                timestamp = timestamp.view(32, 32)
            else:
                raise RuntimeError(f"Invalid tensor size for id_seq or data_seq")

            # Create 32x32x3 tensor
            combined_tensor = torch.stack([id_seq, data_seq, timestamp], dim=-1)
            combined_tensor = combined_tensor.permute(2, 0, 1)  # [C, H, W]

            # Append data and label
            data.append(combined_tensor.numpy())  # Convert to numpy for SMOTE
            labels.append(label[0][0].item())

        return np.array(data), np.array(labels)

    def apply_smote_to_data(self):
        """Apply SMOTE to balance the training data."""
        data_flattened = self.data.reshape(len(self.data), -1)  # Flatten 3D tensor to 2D
        smote = SMOTE(random_state=42)
        data_resampled, labels_resampled = smote.fit_resample(data_flattened, self.labels)

        # Reshape data back to original shape
        self.data = data_resampled.reshape(-1, *self.data.shape[1:])
        self.labels = labels_resampled

        # Update total size
        self.total_size = len(self.labels)

    def __getitem__(self, idx):
        # For SMOTE-applied data
        if self.data is not None and self.labels is not None:
            combined_tensor = torch.tensor(self.data[idx])
            label = self.labels[idx]
        else:
            # Load data as before
            filenames = '{}/{}.tfrec'.format(self.root_dir, idx)
            index_path = None
            description = {'id_seq': 'int', 'data_seq': 'int', 'timestamp': 'float', 'label': 'int'}
            dataset = TFRecordDataset(filenames, index_path, description)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
            data = next(iter(dataloader))
            
            id_seq, data_seq, timestamp, label = data['id_seq'], data['data_seq'], data['timestamp'], data['label']
            id_seq = id_seq.to(torch.float)
            data_seq = data_seq.to(torch.float)
            timestamp = timestamp.to(torch.float)

            id_seq[id_seq == 0] = -1
            if id_seq.numel() == 1024 and data_seq.numel() == 1024:
                id_seq = id_seq.view(32, 32)
                data_seq = data_seq.view(32, 32)
                timestamp = timestamp.view(32, 32)
            else:
                raise RuntimeError(f"Invalid tensor size for id_seq or data_seq")

            # Create 32x32x3 tensor
            combined_tensor = torch.stack([id_seq, data_seq, timestamp], dim=-1)
            combined_tensor = combined_tensor.permute(2, 0, 1)  # [C, H, W]

        # Apply transformations if provided
        if self.transform:
            combined_tensor = self.transform(combined_tensor)
        
        # Ensure label is integer
        if isinstance(label, torch.Tensor):
            label = label.item()  # Convert tensor to integer
        return combined_tensor, label

    def __len__(self):
        return self.total_size
