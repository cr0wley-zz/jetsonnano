import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import csv
import argparse
import datetime
import os
from jtop import jtop, JtopException
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training with jtop Logger')
    parser.add_argument('--file', action="store", dest="file", default="cnnlog.csv")
    args = parser.parse_args()

    # Get the start date and time for the filename
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Specify a different directory
    directory = "./CNNlogs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the full path with the timestamp
    file_path = os.path.join(directory, f"cnnlog_{start_time}.csv")

    print("Training MNIST with jtop logging")
    print(f"Saving log on {file_path}")

    try:
        with jtop() as jetson:
            with open(file_path, 'w') as csvfile:
                stats = jetson.stats
                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                writer.writeheader()

                for epoch in range(5):  # Run 5 epochs
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)

                        optimizer.zero_grad()

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()

                        # Log every 10 batches
                        if i % 10 == 9:
                            stats = jetson.stats
                            writer.writerow(stats)
                            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}')
                            running_loss = 0.0
                            print(f"Log at {stats['time']}")

                # Save the trained model weights
                weights_path = os.path.join(directory, f"model_weights.pth")
                torch.save(model.state_dict(), weights_path)
                print(f"Model weights saved as {weights_path}")

                # Get the end date and time for the filename
                end_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                new_file_path = os.path.join(directory, f"cnnlog_{start_time}_to_{end_time}.csv")
                os.rename(file_path, new_file_path)
                print(f"Log file saved as {new_file_path}")

    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Training interrupted with CTRL-C")
    except IOError:
        print("I/O error")