from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torchinfo import summary
from vae_128_res_pytorch import VAE, train
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class CustomCelebADataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure RGB

        if self.transform:
            image = self.transform(image)

        return image

if __name__ == "__main__":
    data_dir = '../data/celeba/img_align_celeba/'  # Correct relative path
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    print(f"Dataset root: {data_dir}")
    celeba_dataset = CustomCelebADataset(img_dir=data_dir, transform=transform)
    dataloader = DataLoader(celeba_dataset, batch_size=64, shuffle=True)

    data_batch = next(iter(dataloader))

    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    accelerator = Accelerator() # initialize accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader) #prepare model

    unwrapped_model = accelerator.unwrap_model(model)
    summary(unwrapped_model.encoder, input_size=(1, 3, 128, 128))
    summary(unwrapped_model.decoder, input_size=(1, 32, 32, 32))

    train(model, dataloader,optimizer, epochs=100, accelerator=accelerator, save_every=5)