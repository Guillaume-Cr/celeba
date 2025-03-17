import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from torchinfo import summary

## Define my encoder model

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)  # Xavier/Glorot normal initialization
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Initialize biases to zero

class SelfAttention(nn.Module):
    def __init__(self, n_heads, embd_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.in_proj = nn.Linear(embd_dim, 3 * embd_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embd_dim, embd_dim, bias=out_proj_bias)
        self.d_heads = embd_dim // n_heads

    def forward(self, x, casual_mask=False):
        batch_size, seq_len, d_embed = x.shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape)
        k = k.view(interim_shape)
        v = v.view(interim_shape)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        if casual_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape((batch_size, seq_len, d_embed))
        output = self.out_proj(output)
        return output

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residual = x.clone()
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x += residual
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x.clone()
        x = self.groupnorm1(x)
        x = F.selu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = self.conv2(x)
        return x + self.residual_layer(residue)


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

class EncoderModel(nn.Sequential):
    def __init__(self):
        super().__init__(

            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x):
        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        x = mean + eps * std
        x *= 0.18215
        return mean, log_variance, x


class DecoderModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )


    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def forward(self, x):
        mean, variance, encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded, mean, variance

def train_step(model, optimizer, x):
    # Zero the gradients
    optimizer.zero_grad()

    #forward
    z_mean, z_log_var, z = model.encoder.forward(x)
    output = model.decoder.forward(z)

    # Define KL loss
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

    # Define the binary cross entropy
    reconstruction_loss = 2000 * F.mse_loss(output, x, reduction='mean')

    total_loss = kl_loss + reconstruction_loss

    total_loss.backward()

    optimizer.step()

    return total_loss.item(), reconstruction_loss.item(), kl_loss.item()

def loss_fn(recon_x, x, mu, log_var):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence between the learned distribution and the standard normal distribution
    # We use mean and log_variance from encoder
    # Latent variable z = mu + std * eps
    KL = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KL

def train(model, dataloader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.cuda()
            optimizer.zero_grad()
            # Forward pass
            decoded, encoded, mean, variance = model(data)
            # Compute loss
            loss = loss_fn(decoded, data, mean, variance)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(dataloader)}')

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
        transforms.Resize((64, 64)),
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
    summary(model.decoder, input_size=(1, 4, 4, 4))

    train(model, dataloader, epochs=10)
