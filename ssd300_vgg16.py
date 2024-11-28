import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import cv2
import numpy as np

# Define the image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the Pascal VOC dataset
dataset = VOCDetection(root='path/to/VOCdevkit', year='2007', image_set='train', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Load the pre-trained SSD model
model = detection.ssd300_vgg16(pretrained=True)
model.eval()

# Define the training loop
def train_model(model, dataloader, num_epochs=10):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    criterion = detection.SSD300Loss()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

    print('Training complete')

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_model(model, dataloader)
