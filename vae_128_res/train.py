from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torchinfo import summary
from vae_128_res import VAE, train
import torch
import os

if __name__ == "__main__":
    data_dir = '../data'  # Correct relative path
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    print(f"Dataset root: {data_dir}")
    print(f"Contents of data/celeba: {os.listdir(os.path.join(data_dir, 'celeba'))}")
    celeba_dataset = datasets.CelebA(
        root=data_dir,
        split='all',
        target_type='attr',
        transform=transform,
        download=False
    )
    dataloader = DataLoader(celeba_dataset, batch_size=64, shuffle=True)

    data_batch, labels_batch = next(iter(dataloader))

    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    accelerator = Accelerator() # initialize accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader) #prepare model

    unwrapped_model = accelerator.unwrap_model(model)
    summary(unwrapped_model.encoder, input_size=(1, 3, 128, 128))
    summary(unwrapped_model.decoder, input_size=(1, 4, 4, 4))

    train(model, dataloader,optimizer, epochs=100, accelerator=accelerator, save_every=5)