import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import xmltodict
import os
from matplotlib import pyplot as plt

# Define the dataset class
class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.annotation_files = [f.replace('.jpg', '.xml') for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        ann_path = os.path.join(self.root_dir, self.annotation_files[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(ann_path) as f:
            ann = xmltodict.parse(f.read())

        boxes = []
        labels = []
        for obj in ann['annotation']['object']:
            bbox = obj['bndbox']
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Assuming all objects are of the same class

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transform:
            image = self.transform(image)

        return image, target

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the dataset
dataset = BrainDataset(root_dir='__PATH__', transform=transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Load the RetinaNet model
model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()

# Training loop with loss and accuracy tracking
def train_model(model, data_loader, num_epochs=10):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss = []
    train_accuracy = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += sum(len(t['labels']) for t in targets)
            correct += sum((predicted == t['labels']).sum().item() for t in targets)
        
        train_loss.append(running_loss / len(data_loader))
        train_accuracy.append(100. * correct / total)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss[-1]}, Accuracy: {train_accuracy[-1]}%')

    print('Training complete')

    # Plotting the loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training Accuracy')

    plt.show()

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_model(model, data_loader)
