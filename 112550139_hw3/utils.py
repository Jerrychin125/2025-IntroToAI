from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name
    
def load_train_dataset(path: str='/home/jerrychin/NYCU/2025-IntroToAI/112550139_hw3/data/train')->Tuple[List, List]:
    # (TODO) Load training dataset from the given path, return images and labels
    label_map = {"elephant": 0, "jaguar": 1, "lion": 2, "parrot": 3, "penguin": 4}
    images = []
    labels = []
    
    for root, dirs, files in os.walk(path):
        class_name = os.path.basename(root)
        if class_name not in label_map:
            continue
        lbl = label_map[class_name]
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                images.append(os.path.join(root, f))
                labels.append(lbl)
    
    # raise NotImplementedError
    return images, labels

def load_test_dataset(path: str='/home/jerrychin/NYCU/2025-IntroToAI/112550139_hw3/data/test')->List:
    # (TODO) Load testing dataset from the given path, return images
    images = []
    
    for f in os.listdir(path):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            images.append(os.path.join(path, f))
    images.sort()
    
    # raise NotImplementedError
    return images

def plot(train_losses: List, val_losses: List):
    # (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    #        xlabel: 'Epoch', ylabel: 'Loss'
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.png")
    
    # raise NotImplementedError
    print("Save the plot to 'loss.png'")
    return