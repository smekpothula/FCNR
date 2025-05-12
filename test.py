import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from model import FCNRImproved
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_path = 'data/1/VORTS_0001_0.000000_20.905157_0.356822_0.000000_0.934172.png'
checkpoint_path = 'checkpoint.pth'  # trained model checkpoint

model = FCNRImproved().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

try:
    angle_str = image_path.split('_')[-1].replace('.png', '')
    angle = float(angle_str)
except ValueError:
    angle = 0.0  # fallback if filename parsing fails
camera_angles = torch.tensor([[angle, angle]], dtype=torch.float32).to(device)


with torch.no_grad():
    output = model(input_tensor, camera_angles)
    save_image(output, 'reconstructed.png')
    save_image(torch.cat([input_tensor, output], dim=0), 'comparison.png')  # side-by-side


mse = F.mse_loss(output, input_tensor)
psnr = -10 * torch.log10(mse)


bpp = (output.numel() * 8) / (128 * 128 * 3)


print(f"✅ Evaluation Complete")
print(f"PSNR: {psnr.item():.2f} dB")
print(f"BPP: {bpp:.2f}")
from PIL import Image

Image.open('reconstructed.png').show()
Image.open('comparison.png').show()
