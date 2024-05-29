import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import models
import numpy as np
import csv
import argparse
import datetime
import os
from jtop import jtop, JtopException

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")  # To check which device is used

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loader for test dataset
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Define a function to get the model
def get_model(model_name):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 10)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 10)
    else:
        raise ValueError("Model name not recognized. Choose from 'resnet18', 'mobilenet_v2', or 'densenet121'.")
    
    return model.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Inference with jtop Logger')
    parser.add_argument('--model', type=str, required=True, help="Model to use: 'resnet18', 'mobilenet_v2', or 'densenet121'")
    parser.add_argument('--file', type=str, default="inf_log.csv", help="File name for logging")
    args = parser.parse_args()

    # Get the model
    model = get_model(args.model)

    # Get the start date and time for the filename
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Specify a different directory
    directory = "./CNNlogs"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Construct the full path with the timestamp
    file_path = os.path.join(directory, f"cnninf_log_{start_time}.csv")

    # Load the trained model weights
    weights_path = os.path.join(directory, f"model_weights_{args.model}.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Model weights for {args.model} loaded successfully.")

    print(f"Running inference on CIFAR-10 with jtop logging using {args.model}")
    print(f"Saving log on {file_path}")

    try:
        with jtop() as jetson:
            with open(file_path, 'w') as csvfile:
                stats = jetson.stats
                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                writer.writeheader()

                correct = 0
                total = 0

                for i, data in enumerate(test_loader, 0):
                    images, labels = data[0].to(device), data[1].to(device)

                    # Resize images using torch.nn.functional.interpolate for ResNet and DenseNet
                    if args.model != 'mobilenet_v2':
                        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

                    with torch.no_grad():
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                    # Log every 10 batches
                    if i % 10 == 9:
                        stats = jetson.stats
                        writer.writerow(stats)
                        accuracy = 100 * correct / total
                        print(f'Batch {i+1}, Accuracy: {accuracy:.2f} %')
                        print(f"Log at {stats['time']}")

                final_accuracy = 100 * correct / total
                print(f'Accuracy of the network on the 10000 test images: {final_accuracy:.2f} %')

                # Get the end date and time for the filename
                end_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                new_file_path = os.path.join(directory, f"cnninf_log_{args.model}_{start_time}_to_{end_time}.csv")
                os.rename(file_path, new_file_path)
                print(f"Log file saved as {new_file_path}")

    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Inference interrupted with CTRL-C")
    except IOError:
        print("I/O error")