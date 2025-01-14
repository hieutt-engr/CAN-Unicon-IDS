import os
import random
import torch
import time
from torch.utils.data import Dataset
from tfrecord.torch.dataset import TFRecordDataset

class FewShotDataset(Dataset):
    def __init__(self, root_dir, n_way, k_shot, window_size, num_classes, imbalanced_classes=None, transform=None):
        """
        Dataset for few-shot learning directly from .tfrec files with support for prioritizing imbalanced classes.

        Args:
            root_dir: Directory containing .tfrec files.
            n_way: Number of classes per batch.
            k_shot: Number of samples per class.
            window_size: Size of each data window.
            num_classes: Total number of classes.
            imbalanced_classes: List of class indices that are imbalanced (optional).
            transform: Transformations to apply to the data.
        """
        # self.root_dir = root_dir
        self.root_dir = os.path.join(root_dir, 'train')    
        self.n_way = n_way
        self.k_shot = k_shot
        self.window_size = window_size
        self.num_classes = num_classes
        self.imbalanced_classes = imbalanced_classes if imbalanced_classes else []
        self.transform = transform

        # Group indices by label
        self.class_indices = {c: [] for c in range(num_classes)}
        for idx in range(len(os.listdir(self.root_dir))):
            file_path = os.path.join(self.root_dir, f'{idx}.tfrec')
            dataset = TFRecordDataset(file_path, index_path=None, description={
                'id_seq': 'int', 'data_seq': 'int', 'timestamp': 'float', 'label': 'int'
            })
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
            data = next(iter(dataloader))
            label = data['label'][0][0].item()
            self.class_indices[label].append(file_path)

    def __len__(self):
        """
        Number of few-shot episodes.
        """
        return sum(len(files) for files in self.class_indices.values())

    def __getitem__(self, idx):
      """
      Generate a few-shot batch.

      Returns:
          combined_batch: Tensor of shape [n_way * k_shot, C, H, W].
          combined_labels: Tensor of shape [n_way * k_shot].
      """
      # Prioritize imbalanced classes if specified
      # print('Creating batch...')
      # print('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
      if self.imbalanced_classes:
          # Tạo danh sách lớp với trọng số ưu tiên cho các lớp mất cân bằng
          weighted_classes = (
              self.imbalanced_classes * 3  # Ưu tiên lớp mất cân bằng hơn
              + [c for c in self.class_indices if c not in self.imbalanced_classes]
          )
          selected_classes = random.sample(
              weighted_classes,
              self.n_way
          )
      else:
          selected_classes = random.sample(
              [c for c in self.class_indices if len(self.class_indices[c]) > 0],
              self.n_way
          )

      combined_batch = []
      combined_labels = []

      for cls in selected_classes:
          file_paths = self.class_indices[cls]

          # Nếu không đủ mẫu, oversampling
          if len(file_paths) < self.k_shot:
              file_paths = file_paths * (self.k_shot // len(file_paths) + 1)

          selected_files = random.sample(file_paths, self.k_shot)

          for file_path in selected_files:
              dataset = TFRecordDataset(
                  file_path, index_path=None, description={
                      'id_seq': 'int', 'data_seq': 'int', 'timestamp': 'float', 'label': 'int'
                  }
              )
              dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
              try:
                  data = next(iter(dataloader))
              except StopIteration:
                  continue

              id_seq, data_seq, timestamp = data['id_seq'], data['data_seq'], data['timestamp']
              id_seq = id_seq.to(torch.float).view(32, 32)
              data_seq = data_seq.to(torch.float).view(32, 32)
              timestamp = timestamp.to(torch.float).view(32, 32)

              # Combine into [C, H, W]
              combined_tensor = torch.stack([id_seq, data_seq, timestamp], dim=0)

              combined_batch.append(combined_tensor)
              combined_labels.append(cls)

      # Chuyển batch thành tensor
      combined_batch = torch.stack(combined_batch, dim=0)
      combined_labels = torch.tensor(combined_labels, dtype=torch.long)
      # print('End time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
      # print(f"Selected classes: {selected_classes}")
      # print(f"Batch size: {combined_batch.size()}")
      # print(f"Batch labels: {combined_labels}")
      return combined_batch, combined_labels