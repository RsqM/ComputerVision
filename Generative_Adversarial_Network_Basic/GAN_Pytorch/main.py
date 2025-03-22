import os
import torch
import torch.nn as nn
import torch.optim as optim
from gan import Generator, Discriminator
from data_loader import get_data_loader
from utils import weights_init

# Hyperparameters
batch_size = 64
lr = 0.0002
num_epochs = 10
z_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory to save models
os.makedirs('models', exist_ok=True)

# Load data
data_loader = get_data_loader(batch_size)

# Initialize models
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)

# Apply weight initialization
generator.apply(weights_init)
discriminator.apply(weights_init)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(data_loader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_outputs = discriminator(real_images)
        d_loss_real = criterion(real_outputs, real_labels)

        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)

        g_loss.backward()
        optimizer_G.step()

        if i % int(len(data_loader)/10) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    # Save model checkpoints at every 5th epoch
    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), f'models/generator_epoch_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'models/discriminator_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved at epoch {epoch+1}")

print("Training complete!")
