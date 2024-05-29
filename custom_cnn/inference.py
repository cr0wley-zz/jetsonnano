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

# Check for a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net().to(device)

# Load the trained model weights
directory = "./CNNlogs"
if not os.path.exists(directory):
    os.makedirs(directory)
weights_path = os.path.join(directory, f"model_weights.pth")
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Inference with jtop Logger')
    parser.add_argument('--file', action="store", dest="file", default="inf_log.csv")
    args = parser.parse_args()

    # Get the start date and time for the filename
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Specify a different directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Construct the full path with the timestamp
    file_path = os.path.join(directory, f"cnninf_log_{start_time}.csv")

    print("Running inference on MNIST with jtop logging")
    print(f"Saving log on {file_path}")

    try:
        with jtop() as jetson:
            with open(file_path, 'w') as csvfile:
                stats = jetson.stats
                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                writer.writeheader()

                total_loss = 0
                total_accuracy = 0
                total_samples = 0

                for i, data in enumerate(testloader, 0):
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)

                    with torch.no_grad():
                        outputs = model(images)
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)
                        correct = (predicted == labels).sum().item()

                        total_loss += loss.item()
                        total_accuracy += correct
                        total_samples += labels.size(0)

                    # Log every 10 batches
                    if i % 10 == 9:
                        stats = jetson.stats
                        writer.writerow(stats)
                        avg_loss = total_loss / (i + 1)
                        avg_accuracy = total_accuracy / total_samples
                        print(f'Batch {i+1}, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}')
                        print(f"Log at {stats['time']}")

                # Get the end date and time for the filename
                end_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                new_file_path = os.path.join(directory, f"cnninf_log_{start_time}_to_{end_time}.csv")
                os.rename(file_path, new_file_path)
                print(f"Log file saved as {new_file_path}")

    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Inference interrupted with CTRL-C")
    except IOError:
        print("I/O error")