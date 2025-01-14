import torch

class FewShotBatchCreator:
    def __init__(self, dataset, n_way, k_shot):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.few_shot_batch = self._create_few_shot_batch()

    def _create_few_shot_batch(self):
        few_shot_batch = []
        class_indices = self.dataset.class_indices  # Lấy class_indices đã lưu trong dataset

        for c in range(self.n_way):
            if c not in class_indices or len(class_indices[c]) < self.k_shot:
                print(f"Class {c} does not have enough samples. Skipping...")
                continue
            selected_indices = torch.randperm(len(class_indices[c]))[:self.k_shot]
            for idx in selected_indices:
                data, label = self.dataset[class_indices[c][idx]]
                few_shot_batch.append((data, label))

        print(f"Few-Shot Batch Created with {len(few_shot_batch)} samples")
        return few_shot_batch

    def get_batch(self):
        return self.few_shot_batch
