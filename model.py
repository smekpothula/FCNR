import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions

    def forward(self, angles):  # angles: [B, 2]
        encodings = [angles]
        for i in range(1, self.num_encoding_functions + 1):
            encodings.append(torch.sin((2 ** i) * angles))
            encodings.append(torch.cos((2 ** i) * angles))
        return torch.cat(encodings, dim=-1)


class SingleImageEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=4, dilation=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class HierarchicalQuantizer(nn.Module):
    def __init__(self, output_size=(16, 16)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return torch.round(self.pool(x) * 255) / 255


class FCNRImproved(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SingleImageEncoder()
        self.quantizer = HierarchicalQuantizer()

        pose_dim = 2 + 2 * 6 * 2
        self.fc1 = nn.Linear(16 * 16 * 256 + pose_dim, 1024)
        self.fc2 = nn.Linear(1024, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Tanh()
        )

        self.pose_encoding = PositionalEncoding()

    def forward(self, image, camera_angles):
        feats = self.encoder(image)
        q = self.quantizer(feats)
        q_flat = q.view(q.size(0), -1)
        pose = self.pose_encoding(camera_angles)
        x = torch.cat([q_flat, pose], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.decoder(x)
        return out