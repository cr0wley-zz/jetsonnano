import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
from jtop import jtop, JtopException
import sys
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else sys.exit(-1))

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        # Calculate the size of the output tensor
        output_size = img_shape[0] * img_shape[1] * img_shape[2]
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Hyperparameters
latent_dim = 100
img_shape = (1, 28, 28)
batch_size = 64
lr = 0.0002
betas = (0.5, 0.999)
epochs = 100

# Initialize generator and discriminator
generator = Generator(latent_dim, img_shape).to(device)
discriminator = Discriminator(img_shape[0]*img_shape[1]*img_shape[2]).to(device)

# Loss function and optimizer
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




# Open CSV file for writing
with jtop() as jetson:
    with open('jetson_stats.csv', mode='w', newline='') as file:
        stats=jetson.stats
        fieldnames=list(stats.keys())
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        writer.writeheader()
        # Training loop
        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(dataloader):

                # Adversarial ground truths
                valid = torch.ones(imgs.size(0), 1).to(device)
                fake = torch.zeros(imgs.size(0), 1).to(device)

                # Configure input
                real_imgs = imgs.view(imgs.size(0), -1).to(device)

                # -----------------
                # Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = torch.randn(imgs.size(0), latent_dim).to(device)

                # Generate a batch of images
                gen_imgs = generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs.view(imgs.size(0), -1)), valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                # Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()
                # Write Jetson stats to CSV
                try:

                    writer.writerow(jetson.stats)
                except KeyError:
                    pass
                    

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, epochs, i+1, len(dataloader), d_loss.item(), g_loss.item())
            )