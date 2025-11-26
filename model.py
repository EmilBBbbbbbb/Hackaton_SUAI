import os
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.models as models

from generate_dataset import SyntheticMaskDataset
import numpy as np
from transform import order_points, four_point_transform
import matplotlib.pyplot as plt

class CornerNet(nn.Module):

    def __init__(self, pretrained=False, dropout=0.3):
        super().__init__()
        m = models.resnet18(pretrained=pretrained)

        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = nn.Sequential(*list(m.children())[:-1])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.encoder(x)
        out = self.head(f)
        return out

def train_corner_net(device='cuda' if torch.cuda.is_available() else 'cpu',
                     epochs=15, batch_size=16, lr=1e-3, samples=8000, val_split=0.1,
                     out_size=512, save_path='models/corner_net.pth'):

    # 1. Dataset
    dataset = SyntheticMaskDataset(n_samples=samples, out_size=out_size)
    val_n = int(len(dataset) * val_split)
    train_n = len(dataset) - val_n
    train_set, val_set = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = CornerNet(pretrained=False).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')

    for epoch in range(1, epochs+1):
        model.train()
        running_train = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_train += loss.item() * x.size(0)

        train_loss = running_train / train_n

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                running_val += loss.item() * x.size(0)

        val_loss = running_val / val_n
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict(), 'out_size': out_size}, save_path)
            print(f"Saved best model: {save_path}")

    return model


def predict_corners_from_mask(model, mask, device='cpu', out_size=512):
    H_orig, W_orig = mask.shape[:2]

    m = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    if m.max() > 1:
        m = m / 255.0

    inp = torch.from_numpy(m).float().to(device)[None, None]

    with torch.no_grad():
        pred = model(inp)[0].cpu().numpy()
    pred = pred.reshape(4, 2)

    pred[:, 0] *= (W_orig - 1)
    pred[:, 1] *= (H_orig - 1)

    return pred

if __name__ == "__main__":
    MODEL_PATH = "models/corner_net.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(MODEL_PATH):
        print("Training model...")
        train_corner_net(
            device=device,
            epochs=10,
            batch_size=16,
            lr=1e-3,
            samples=4000,
            out_size=512,
            save_path=MODEL_PATH
        )

    chk = torch.load(MODEL_PATH, map_location=device)
    model = CornerNet(pretrained=False).to(device)
    model.load_state_dict(chk['model_state'])
    out_size = chk.get('out_size', 512)

    mask = cv2.imread('exemple_docs/ex1_mask.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('exemple_docs/ex1.jpg')

    corners = predict_corners_from_mask(model, mask, device=device, out_size=out_size)

    pts = np.array(corners, dtype='float32')
    pts = order_points(pts)
    warped = four_point_transform(image, pts)

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(image_rgb)
        ax1.set_title('Original')
        ax1.axis('off')
        ax2.imshow(warped_rgb)
        ax2.set_title('Warped')
        ax2.axis('off')
        plt.show()
    except Exception as e:
        print("Visualization skipped:", e)