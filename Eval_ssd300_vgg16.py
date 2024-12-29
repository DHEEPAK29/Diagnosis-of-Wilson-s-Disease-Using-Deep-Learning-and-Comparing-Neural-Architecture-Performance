import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import cv2
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt

# Ensure MATLAB's interface to Python is available
import matlab.engine

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

# Save the detection results to a .mat file
def save_detection_results(filename, boxes, labels, scores):
    # Create a dictionary to hold the data
    detection_data = {
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    }
    
    # Save the dictionary to a .mat file
    sio.savemat(filename, detection_data)

# Save results after evaluating the model
save_detection_results('detection_results.mat', boxes, labels, scores)

# Optionally, visualize the results using OpenCV (before MATLAB visualization)
for i in range(len(boxes)):
    image = cv2.imread(dataset.images[i])
    for box, label, score in zip(boxes[i], labels[i], scores[i]):
        if score > 0.5:  # Threshold for visualization
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow(f"Image {i+1}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Now use MATLAB for visualization via Python-MATLAB interface
def visualize_in_matlab():
    # Start the MATLAB engine
    eng = matlab.engine.start_matlab()

    # Load the detection results into MATLAB
    data = eng.load('detection_results.mat')

    boxes = data['boxes']
    labels = data['labels']
    scores = data['scores']

    # Visualization in MATLAB
    for i in range(len(boxes)):
        # Load the image in MATLAB (ensure path is correct)
        image_path = dataset.images[i]
        image = eng.imread(image_path)

        # Convert image to RGB (in case it's not already in that format)
        image_rgb = eng.ind2rgb(image, [0, 0, 0])

        # Create figure for visualization
        eng.figure()

        # Loop through the detections and plot them
        for j in range(len(boxes[i])):
            box = boxes[i][j]
            label = labels[i][j]
            score = scores[i][j]

            if score > 0.5:  # Only show detections with score > 0.5
                # Draw the bounding box
                eng.rectangle('Position', [box[0], box[1], box[2]-box[0], box[3]-box[1]], 'EdgeColor', 'g', 'LineWidth', 2)
                eng.text(box[0], box[1] - 10, f'{label}: {score:.2f}', 'Color', 'g', 'FontSize', 12)

        # Display the image
        eng.imshow(image_rgb)
        eng.title(f"Image {i + 1} - Detections")

        # Pause to view each image
        eng.pause(1)  # Adjust the pause time as necessary

    # Close the MATLAB engine
    eng.quit()

# Call the MATLAB visualization function
visualize_in_matlab()
