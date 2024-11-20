import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import xmltodict
import os
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

# Define the dataset class (same as before)
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
dataset = BrainDataset(root_dir='path/to/your/dataset', transform=transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Load the RetinaNet model
model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
model.eval()

# Function to convert predictions to COCO format
def convert_to_coco_format(outputs, image_ids):
    coco_results = []
    for i, output in enumerate(outputs):
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        for j in range(len(boxes)):
            box = boxes[j]
            score = scores[j]
            label = labels[j]
            coco_results.append({
                'image_id': image_ids[i],
                'category_id': label,
                'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                'score': score
            })
    return coco_results

# Evaluate the model
coco_gt = COCO('path/to/annotations.json')  # Load ground truth annotations in COCO format
coco_dt = coco_gt.loadRes(convert_to_coco_format(outputs, image_ids))  # Load model predictions

coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Print evaluation results
print('mAP:', coco_eval.stats[0])
