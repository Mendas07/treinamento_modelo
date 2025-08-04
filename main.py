import os
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Dataset usando arquivos CSV com os rótulos
class MNISTData(Dataset):
    def __init__(self, image_dir, labels_csv, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        label = int(self.labels_df.iloc[idx, 1])
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label

# Transformações
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Hiperparâmetros
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 0.001

# Carrega os datasets
train_dataset = MNISTData("mnist data/train", "mnist data/train_labels.csv", transform=transform)
test_dataset = MNISTData("mnist data/test", "mnist data/test_labels.csv", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Visualizar um exemplo
sample_img, sample_label = train_dataset[0]
plt.imshow(sample_img.permute(1, 2, 0))
plt.title(f"Label: {sample_label}")
plt.show()

# Modelo ResNet18
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 10)  # MNIST tem 10 classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Função de treino
def train(model, loader):
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss:.4f}")
