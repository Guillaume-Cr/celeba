import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

import wandb

from tqdm import tqdm

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
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=1, padding=0)
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
            nn.Conv2d(32, 512, kernel_size=3, padding=1),
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

def train(model, dataloader, optimizer, epochs=100, accelerator=None, save_every=1, save_dir="./saved_models"):

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    if accelerator is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)
    wandb.init(project="vae-celeba-pytorch-res")

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        recon_loss = 0
        kl_loss = 0

        if accelerator is None:
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{epochs}]")
        else:
            total_batches = len(dataloader) * accelerator.num_processes if accelerator is not None else len(dataloader)
            progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch [{epoch+1}/{epochs}]")

        for batch_idx, data in progress_bar:
            if accelerator is None:
                data = data.cuda()
            optimizer.zero_grad()
            # Forward pass
            decoded, encoded, mean, variance = model(data)
            # Compute loss
            BCE = F.mse_loss(decoded, data, reduction='sum')
            # KL Divergence between the learned distribution and the standard normal distribution
            # We use mean and log_variance from encoder
            # Latent variable z = mu + std * eps
            KL = -0.5 * torch.sum(1 + variance - mean.pow(2) - variance.exp())
            loss = 2000000 * BCE + KL

            if accelerator is None:
                progress_bar.set_postfix({"Loss": loss.item()})
            else:
                progress_bar.set_postfix({"Loss": accelerator.gather(loss).mean().item()})
            if accelerator is None:
                loss.backward()
                optimizer.step()
            else:
                accelerator.backward(loss)
                optimizer.step()

            if accelerator is None:
                train_loss += loss.item()
                recon_loss += BCE.item()
                kl_loss += KL.item()
            else:
                train_loss += accelerator.gather(loss).mean().item()
                recon_loss += accelerator.gather(BCE).mean().item()
                kl_loss += accelerator.gather(KL).mean().item()

        avg_loss = train_loss/len(dataloader)
        avg_recon = recon_loss/len(dataloader)
        avg_kl = kl_loss/len(dataloader)
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_recon_loss": avg_recon,
            "avg_kl_loss": avg_kl,
        })

        model.eval()
        with torch.no_grad():
            sample_data, _ = next(iter(dataloader))
            if accelerator is None:
                sample_data = sample_data.cuda()
            unwrapped_model = accelerator.unwrap_model(model)
            _, _, z = unwrapped_model.encoder(sample_data[:10])
            reconstructed_images = unwrapped_model.decoder(z).cpu()
            display(reconstructed_images, save_to=f"reconstructed_epoch_{epoch}.png")
            wandb.log({f"reconstructed_images_epoch_{epoch}": wandb.Image(f"reconstructed_epoch_{epoch}.png")})
        model.train()

        # Save model at fixed checkpoints
        if accelerator is None:
            torch.save(model.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch + 1}.pth"))
            print(f"Saved model at epoch {epoch + 1}")
        else:
            unwrapped_model = accelerator.unwrap_model(model) # Get the original model
            if accelerator.is_main_process: # Only save on the main process
                torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch + 1}.pth"))
                print(f"Saved model at epoch {epoch + 1}")

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(dataloader)}')
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

