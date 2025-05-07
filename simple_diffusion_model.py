import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F # Use torch.nn.functional for padding
import json
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
DATASET_PATH_DATA = 'minecraft_textures_dataset.npy'
DATASET_PATH_LABELS = 'minecraft_textures_labels.npy'
LABEL_MAP_PATH = 'minecraft_label_map.json'
IMAGE_SIZE = 16
NUM_CHANNELS = 3
TIMESTEPS = 1000 # Number of diffusion steps
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
EPOCHS = 1500 # Increased epochs for better training
TIME_EMB_DIM = 32
LABEL_EMB_DIM = 32 # Dimension for label embedding
MODEL_SAVE_PATH = 'simple_diffusion_model.pth' # Path to save the trained model

# --- Diffusion Process Helper Functions ---

def linear_beta_schedule(timesteps):
    """
    Linear schedule for beta values.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def extract(a, t, x_shape):
    """
    Extract the appropriate t index for a batch of indices.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Define beta schedule
betas = linear_beta_schedule(TIMESTEPS)

# Define alphas and related quantities
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), "constant", 1.0) # Use constant padding with value
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1. / alphas)
variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# --- Model Definition (Simple U-Net) ---

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.exp(torch.arange(half_dim, device=device) * - (torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)))
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.mlp(embeddings)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, label_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.label_mlp = nn.Linear(label_emb_dim, out_channels) # Label embedding projection

        # Residual connection projection if channels don't match
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity() # Use Identity if channels match

    def forward(self, x, time_emb, label_emb):
        residual = self.residual_conv(x) # Apply projection to input for residual

        h = self.relu(self.conv1(x))
        # Add time and label embeddings
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        if label_emb is not None: # Add label embedding only if provided
             h = h + self.label_mlp(label_emb)[:, :, None, None]
        h = self.relu(self.conv2(h))

        # Add residual connection
        h = h + residual

        return h

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS, time_emb_dim=TIME_EMB_DIM, num_classes=None, label_emb_dim=LABEL_EMB_DIM):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)

        if num_classes is not None:
            self.label_embedding = nn.Embedding(num_classes, label_emb_dim)
        else:
            self.label_embedding = None

        # Increased channels for a larger model
        self.inc = Block(in_channels, 128, time_emb_dim, label_emb_dim)
        self.down1 = Downsample(128, 256)
        self.block1 = Block(256, 256, time_emb_dim, label_emb_dim)
        self.down2 = Downsample(256, 512)
        self.block2 = Block(512, 512, time_emb_dim, label_emb_dim)

        self.mid = Block(512, 512, time_emb_dim, label_emb_dim)

        self.up1 = Upsample(512, 256)
        self.block3 = Block(256 + 256, 256, time_emb_dim, label_emb_dim) # 256 (from upsample) + 256 (from skip connection)
        self.up2 = Upsample(256, 128)
        self.block4 = Block(128 + 128, 128, time_emb_dim, label_emb_dim) # 128 (from upsample) + 128 (from skip connection)

        self.outc = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x, t, labels=None):
        time_emb = self.time_embedding(t)
        label_emb = None
        if self.label_embedding is not None and labels is not None:
            label_emb = self.label_embedding(labels)

        # Downsampling path
        x1 = self.inc(x, time_emb, label_emb)
        x2 = self.down1(x1)
        x2 = self.block1(x2, time_emb, label_emb)
        x3 = self.down2(x2)
        x3 = self.block2(x3, time_emb, label_emb)

        # Bottleneck
        x3 = self.mid(x3, time_emb, label_emb)

        # Upsampling path with skip connections
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1) # Skip connection
        x = self.block3(x, time_emb, label_emb)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1) # Skip connection
        x = self.block4(x, time_emb, label_emb)

        # Output layer
        output = self.outc(x)
        return output

# --- Data Loading and Preprocessing ---

class TextureDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        # Normalize pixel values from [0, 255] to [-1, 1]
        self.data = self.data.astype(np.float32) / 127.5 - 1
        # Convert numpy array to torch tensor and change shape from (N, H, W, C) to (N, C, H, W)
        self.data = torch.from_numpy(self.data).permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(self.labels).long() # Labels as LongTensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_dataset(data_path, label_path):
    """
    Loads the dataset and labels.
    """
    dataset = TextureDataset(data_path, label_path)
    return dataset

# --- Training Function ---

def train(model, dataloader, optimizer, epochs, timesteps, device):
    model.train()
    for epoch in range(epochs):
        for step, (batch, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.to(device)
            labels = labels.to(device)
            batch_size = batch.shape[0]

            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            # Add noise to the images
            noise = torch.randn_like(batch)
            sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, batch.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, batch.shape)
            noisy_images = sqrt_alphas_cumprod_t * batch + sqrt_one_minus_alphas_cumprod_t * noise

            # Predict the noise
            predicted_noise = model(noisy_images, t, labels)

            # Calculate loss (MSE between predicted noise and actual noise)
            loss = nn.functional.mse_loss(predicted_noise, noise)

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Step {step}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch}/{epochs} finished, Loss: {loss.item():.4f}")

# --- Sampling Function ---

@torch.no_grad()
def sample(model, shape, timesteps, device, labels=None):
    """
    Generates new images using the reverse diffusion process.
    Optionally condition on labels.
    """
    model.eval()
    img = torch.randn(shape, device=device) # Start with random noise

    if labels is not None:
        labels = labels.to(device)

    for i in reversed(range(timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)

        sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, shape)
        betas_t = extract(betas, t, shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, shape)

        # Predict noise (with conditioning if labels are provided)
        predicted_noise = model(img, t, labels)

        # Calculate mean of the posterior distribution
        mean = sqrt_recip_alphas_t * (img - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if i > 0:
            # Sample from the posterior distribution
            posterior_variance_t = extract(variance, t, shape)
            noise = torch.randn_like(img)
            img = mean + torch.sqrt(posterior_variance_t) * noise
        else:
            img = mean # No noise added at the last step

    # Denormalize the image from [-1, 1] to [0, 255]
    img = (img + 1) * 127.5
    img = torch.clamp(img, 0, 255).to(torch.uint8)
    return img

# --- Main Execution ---

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and labels
    dataset = load_dataset(DATASET_PATH_DATA, DATASET_PATH_LABELS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} images.")

    # Load label map to get number of classes
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"Loaded label map with {num_classes} classes.")

    # Initialize model, optimizer
    model = SimpleUNet(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("Starting training...")
    train(model, dataloader, optimizer, EPOCHS, TIMESTEPS, device)
    print("Training finished.")

    # Save the trained model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved.")

    # # Sample some images (e.g., 2 of each class)
    # print("Sampling new images...")
    # sample_batch_size_per_class = 2
    # sampled_images = []
    # output_base_dir = 'generated_textures_conditional'
    # os.makedirs(output_base_dir, exist_ok=True)

    # # Create a reverse label map for saving
    # reverse_label_map = {v: k for k, v in label_map.items()}

    # for label_id in range(num_classes):
    #     label_name = reverse_label_map.get(label_id, f'unknown_label_{label_id}')
    #     # Create subdirectory for the current label
    #     output_dir = os.path.join(output_base_dir, label_name.replace('/', '_')) # Replace '/' in label names for valid directory names
    #     os.makedirs(output_dir, exist_ok=True)

    #     print(f"Generating {sample_batch_size_per_class} textures for label '{label_name}' (ID: {label_id})...")
    #     # Create a batch of labels for sampling
    #     sample_labels = torch.full((sample_batch_size_per_class,), label_id, dtype=torch.long)
    #     generated_batch = sample(model, (sample_batch_size_per_class, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), TIMESTEPS, device, labels=sample_labels)

    #     for i in range(sample_batch_size_per_class):
    #         img_array = generated_batch[i].permute(1, 2, 0).cpu().numpy()
    #         img = Image.fromarray(img_array, 'RGB')
    #         img.save(os.path.join(output_dir, f'{label_name.replace("/", "_")}_{i:02d}.png')) # Also replace '/' in filename
    #         print(f"Saved {label_name.replace('/', '_')}_{i:02d}.png")

    # print(f"Generated textures for {num_classes} classes in '{output_dir}' directory.")
