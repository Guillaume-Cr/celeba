import torch
from torchvision import transforms
from PIL import Image
from vae_128_res_pytorch import VAE  # Import your VAE model
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import wandb

def load_model(model_path, device):
    """Loads the VAE model from the given path."""
    model = VAE()
    model.load_state_dict(torch.load(model_path, map_location=device)) #Here is where it is used.
    model.to(device)
    model.eval()
    return model

def process_image(image_path, device):
    """Processes the input image."""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def get_latent_and_reconstruction(model, image_tensor):
    """Gets the latent vector and reconstruction from the model."""
    with torch.no_grad():
        latent, mean, variance = model.encoder(image_tensor)
        reconstructed_image = model.decoder(latent)
    return latent, reconstructed_image

def display_images(original_image, reconstructed_image):
    """Displays the original and reconstructed images."""
    original_image = original_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    reconstructed_image = reconstructed_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title("Reconstructed Image")
    plt.axis('off')

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="VAE Image Processing")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--model_path", type=str, default="saved_models/vae_epoch_5.pth", help="Path to the trained VAE model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
        model = load_model(args.model_path, device)
        image_tensor = process_image(args.image_path, device)
        latent_vector, reconstructed_image = get_latent_and_reconstruction(model, image_tensor)

        print("Latent Vector Shape:", latent_vector.shape)
        display_images(image_tensor, reconstructed_image)

    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()