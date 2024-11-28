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
dataset = VOCDetection(root='path/to/VOCdevkit', year='2007', image_set='val', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

# Load the pre-trained SSD model
model = detection.ssd300_vgg16(pretrained=True)
model.eval()

# Define the evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_boxes = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            for output in outputs:
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                
                all_boxes.append(boxes)
                all_labels.append(labels)
                all_scores.append(scores)
    
    return all_boxes, all_labels, all_scores

# Evaluate the model
boxes, labels, scores = evaluate_model(model, dataloader)

# Print evaluation results
print("Evaluation Results:")
for i in range(len(boxes)):
    print(f"Image {i+1}:")
    print(f"Boxes: {boxes[i]}")
    print(f"Labels: {labels[i]}")
    print(f"Scores: {scores[i]}")

# Optionally, visualize the results
for i in range(len(boxes)):
    image = cv2.imread(dataset.images[i])
    for box, label, score in zip(boxes[i], labels[i], scores[i]):
        if score > 0.5:  # Threshold for visualization
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow(f"Image {i+1}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
