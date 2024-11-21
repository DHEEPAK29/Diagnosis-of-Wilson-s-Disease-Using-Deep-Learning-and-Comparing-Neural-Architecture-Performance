import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import xmltodict
import os
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain

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
dataset = BrainDataset(root_dir='PATH', transform=transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Load the EfficientDet model
config = get_efficientdet_config('tf_efficientdet_d0')
model = EfficientDet(config, pretrained_backbone=True)
model = DetBenchTrain(model, config)
model.eval()

# Training loop (simplified)
for images, targets in data_loader:
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]

    with torch.no_grad():
        output = model(images, targets)
    print(output)
