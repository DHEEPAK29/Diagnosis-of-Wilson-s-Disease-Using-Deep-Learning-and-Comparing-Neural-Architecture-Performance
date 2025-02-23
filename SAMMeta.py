import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

class MyDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)
        annotation_path = os.path.join(self.annotation_dir, image_name.split('.')[0] + '.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            image = self.transforms(image)

        return {
            'image': torch.tensor(image).permute(2, 0, 1).float(),
            'annotation': torch.tensor(annotation).long()
        }


# SAM Meta model architecture
class SAMMetaModel(nn.Module):
    def __init__(self):
        super(SAMMetaModel, self).__init__()
        self.encoder = torchvision.models.resnet50(pretrained=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#  hyperparameters
image_dir = 'PATH'
annotation_dir = 'path'
batch_size = 4
num_epochs = 10
learning_rate = 0.001

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset and create data loaders
dataset = MyDataset(image_dir, annotation_dir, transforms=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and loss function
model = SAMMetaModel()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Train model
for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch['image'].to(device)
        annotations = batch['annotation'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = loss_fn(outputs, annotations)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Checkpoint 
torch.save(model.state_dict(), 'sam_meta_model.pth')


# predictions
def make_predictions(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).to(device)

    output = model(image)
    output = torch.sigmoid(output)
    output = output.squeeze(0).cpu().numpy()

    return output


model.load_state_dict(torch.load('sam_meta_model.pth'))
image_path = 'path/to/image.jpg'
prediction = make_predictions(model, image_path)

# prediction
plt.imshow(prediction, cmap='gray')
plt.show()
