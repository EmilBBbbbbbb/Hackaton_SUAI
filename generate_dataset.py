import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from synthetic_mask_generator import generate_document_mask


class SyntheticMaskDataset(Dataset):
    def __init__(self, n_samples=10000, out_size=512):
        self.n_samples = n_samples
        self.out_size = out_size
        self.to_tensor = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        canvas, corners = generate_document_mask()

        mask_resized = cv2.resize(canvas, (self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)

        # Ensure mask is float32 in range [0, 1] for consistency
        mask_resized = mask_resized.astype(np.float32)

        H_new, W_new = mask_resized.shape
        corners_scaled = corners.copy()
        corners_scaled[:,0] = corners_scaled[:,0] * (W_new / canvas.shape[1])
        corners_scaled[:,1] = corners_scaled[:,1] * (H_new / canvas.shape[0])
        target = corners_scaled / np.array([W_new-1, H_new-1], dtype=np.float32)

        x = self.to_tensor(mask_resized)
        y = torch.from_numpy(target.reshape(-1)).float()

        return x, y

if __name__ == "__main__":
    dataset = SyntheticMaskDataset(n_samples=5, out_size=512)
    for i in range(len(dataset)):
        x, y = dataset[i]
        print(f"Sample {i}:")
        print(f"  Mask shape: {x.shape}, dtype: {x.dtype}, min: {x.min()}, max: {x.max()}")
        print(f"  Target shape: {y.shape}, dtype: {y.dtype}, values: {y.numpy()}")