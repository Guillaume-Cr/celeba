import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchinfo import summary
import wandb

## Define my encoder model

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)  # Xavier/Glorot normal initialization
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Initialize biases to zero

class Sampling(nn.Module):
    """
    Sampling layer for VAE that implements the reparameterization trick
    """
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, z_mean, z_log_var):
        batch_size = z_mean.size(0)
        latent_dim = z_mean.size(1)

        # Sample from standard normal distribution
        epsilon = torch.randn(batch_size, latent_dim, device=z_mean.device)

        # Reparameterization trick: z = μ + σ * ε (where σ = exp(log_var/2))
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class EncoderModel(nn.Module):
    def __init__(self, latent_dim=200):
        super(EncoderModel, self).__init__()

        # Define the CNN layers using Sequential
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Flatten()
        )

        # Latent space
        self.z_mean = nn.Linear(512, latent_dim)
        self.z_log_var = nn.Linear(512, latent_dim)
        self.sampling = Sampling()

    def forward(self, x):
        x = self.encoder_cnn(x)

        # Generate latent parameters
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        # Sample from the distribution
        z = self.sampling(z_mean, z_log_var)

        return z_mean, z_log_var, z

class DecoderModel(nn.Module):
    def __init__(self, latent_dim=200):
        super(DecoderModel, self).__init__()

        # Latent dim conversion
        self.dense = nn.Linear(latent_dim, 512)

        self.batch_norm_dense = nn.BatchNorm1d(512)

        self.leaky_relu_dense = nn.LeakyReLU()

         # Define the CNN layers using Sequential
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1, output_padding=0),
        )


    def forward(self, x):
        # Linear projection and reshape

        x = self.dense(x)
        x = self.batch_norm_dense(x)
        x = self.leaky_relu_dense(x)

        # Reshape to match the expected input for transposed convolutions
        # The shape should match the flattened output of the encoder's last conv layer
        # If encoder output was 2x2x256 before flattening
        x = x.view(-1, 128, 2, 2)  # [-1, channels, height, width]

        # Apply transposed convolutions
        x = self.decoder_conv(x)

        x = torch.sigmoid(x)

        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=200):
        super(VAE, self).__init__()

        self.encoder = EncoderModel(latent_dim)
        self.decoder = DecoderModel(latent_dim)

def train_step(model, optimizer, x):
    # Zero the gradients
    optimizer.zero_grad()

    #forward
    z_mean, z_log_var, z = model.encoder.forward(x)
    print(z)
    output = model.decoder.forward(z)

    # Define KL loss
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

    # Define the binary cross entropy
    reconstruction_loss = 1 * F.mse_loss(output, x, reduction='mean')

    total_loss = kl_loss + reconstruction_loss

    total_loss.backward()

    optimizer.step()

    return total_loss.item(), reconstruction_loss.item(), kl_loss.item()

def train(model, dataloader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    wandb.init(project="vae-celeba-pytorch")

    for epoch in range(epochs):
        #Set the model to training mode (Under the hood, set compute gradients to true)
        model.train()

        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0

        for batch_idx, data in enumerate(dataloader):
            x, _ = data
            x = x.to(device)

            loss, recon_loss, kl_loss = train_step(model, optimizer, x)

            train_loss += loss
            recon_loss_total += recon_loss
            kl_loss_total += kl_loss

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.4f}, '
                      f'Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}')

        avg_loss = train_loss / len(dataloader)
        avg_recon = recon_loss_total / len(dataloader)
        avg_kl = kl_loss_total / len(dataloader)

        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}, '
              f'Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}')
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_recon_loss": avg_recon,
            "avg_kl_loss": avg_kl,
        })

        # Generate and log sample images
        if epoch % 5 == 0: # log every 5 epochs
            model.eval()
            with torch.no_grad():
                sample_data, _ = next(iter(dataloader))
                sample_data = sample_data.to(device)
                _, _, z = model.encoder(sample_data[:10])
                reconstructed_images = model.decoder(z).cpu()
                input_images = sample_data[:10].cpu()
                
                 # Display and log input images
                display(input_images, save_to=f"input_epoch_{epoch}.png")
                wandb.log({f"input_images_epoch_{epoch}": wandb.Image(f"input_epoch_{epoch}.png")})

                # Display and log reconstructed images
                display(reconstructed_images, save_to=f"reconstructed_epoch_{epoch}.png")
                wandb.log({f"reconstructed_images_epoch_{epoch}": wandb.Image(f"reconstructed_epoch_{epoch}.png")})

            model.train()

    wandb.finish()

def display(
    images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None
):
    """
    Displays n random import matplotlib.pyplot as pltimages from each one of the supplied arrays.
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    images = images.transpose(0, 2, 3, 1)

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()

if __name__ == "__main__":
    data_dir = './data'  # Correct relative path
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
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
    dataloader = DataLoader(celeba_dataset, batch_size=128, shuffle=True)

    data_batch, labels_batch = next(iter(dataloader))

    display(data_batch)

    model = VAE()
    init_weights(model)

    summary(model.encoder, input_size=(1, 3, 32, 32))
    summary(model.decoder, input_size=(1, 200))

    train(model, dataloader, epochs=11)
