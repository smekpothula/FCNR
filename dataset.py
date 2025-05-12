import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class VortexDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for sub in sorted(os.listdir(root_dir)):
            subdir = os.path.join(root_dir, sub)
            if not os.path.isdir(subdir):
                continue
            for fname in os.listdir(subdir):
                if not fname.endswith('.png'):
                    continue
                path = os.path.join(subdir, fname)
                # parse last underscore-separated value as angle
                try:
                    angle = float(fname.split('_')[-1].replace('.png', ''))
                except:
                    angle = 0.0
                self.samples.append((path, angle))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, angle = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # duplicate angle for theta and phi
        angles = torch.tensor([angle, angle], dtype=torch.float32)
        return img, angles