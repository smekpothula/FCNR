import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import VortexDataset
from model import FCNRImproved

BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 50

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = VortexDataset('data', transform=data_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FCNRImproved().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for imgs, angles in dataloader:
        imgs = imgs.to(device)
        angles = angles.to(device)
        recon = model(imgs, angles)
        loss = criterion(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{EPOCHS}, Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), 'checkpoint.pth')
print("Model saved as checkpoint.pth")
