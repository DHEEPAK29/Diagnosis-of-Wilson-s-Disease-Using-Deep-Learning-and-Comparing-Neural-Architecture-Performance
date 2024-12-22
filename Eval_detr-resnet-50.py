import os
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import average_precision_score
from collections import defaultdict

# Function to calculate IoU (Intersection over Union)
def calculate_iou(pred_box, gt_box):
    x1, y1, x2, y2 = pred_box
    x1_gt, y1_gt, x2_gt, y2_gt = gt_box

    # Calculate intersection
    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    # Calculate union
    pred_area = (x2 - x1) * (y2 - y1)
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
    union_area = pred_area + gt_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# Function to evaluate on the dataset
def evaluate_model(model, processor, image_dir, annotation_dir, threshold=0.5):
    # Store the results and ground truths
    all_detections = []
    all_ground_truths = []

    # Iterate over all images in the dataset
    for image_name in os.listdir(image_dir):
        if not image_name.endswith(".jpg"):
            continue
        
        image_path = os.path.join(image_dir, image_name)
        annotation_path = os.path.join(annotation_dir, image_name.replace(".jpg", ".xml"))
        
        # Load the image
        image = Image.open(image_path)
        
        # Prepare input for the model
        inputs = processor(images=image, return_tensors="pt")
        
        # Perform object detection
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process the detections
        target_sizes = torch.tensor([image.size[::-1]])  # height, width
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        
        # Parse the ground truth annotations (PASCAL VOC format)
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        gt_boxes = []
        gt_labels = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            gt_boxes.append([xmin, ymin, xmax, ymax])
            gt_labels.append(obj.find("name").text)

        # Append the predictions and ground truths for evaluation
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > threshold:
                prediction = {
                    "score": score.item(),
                    "label": label.item(),
                    "box": box.tolist()
                }
                all_detections.append(prediction)

        for gt_box in gt_boxes:
            ground_truth = {
                "label": "brain_region",  # Adjust if there are specific labels
                "box": gt_box
            }
            all_ground_truths.append(ground_truth)
        
        # Plotting the results
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        
        # Plot ground truth boxes
        for gt_box in gt_boxes:
            xmin, ymin, xmax, ymax = gt_box
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        
        # Plot prediction boxes
        for score, box in zip(results["scores"], results["boxes"]):
            if score > threshold:
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        
        plt.show()
    
    # Calculate mAP (mean Average Precision)
    # This requires predictions and ground truths to be grouped by their labels
    labels = defaultdict(list)
    for pred in all_detections:
        labels[pred["label"]].append(pred)

    # Calculate IoU, precision, recall for each object detected
    iou_threshold = 0.5  # You can change this threshold for IoU

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gt in all_ground_truths:
        gt_box = gt["box"]
        matched = False
        for pred in all_detections:
            pred_box = pred["box"]
            iou = calculate_iou(pred_box, gt_box)
            if iou > iou_threshold:
                matched = True
                true_positive += 1
                break
        if not matched:
            false_negative += 1

    # False positives are those predictions that don't match any ground truth
    false_positive = len(all_detections) - true_positive

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    # Average Precision calculation
    ap = average_precision_score([1]*true_positive + [0]*false_positive, 
                                 [p["score"] for p in all_detections])

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    return precision, recall, ap

# Load pre-trained DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Define your image and annotation directories
image_dir = "path_to_images"
annotation_dir = "path_to_annotations"

# Evaluate the model
evaluate_model(model, processor, image_dir, annotation_dir)

# import os
# import torch
# from transformers import DetrImageProcessor, DetrForObjectDetection
# from PIL import Image
# import xml.etree.ElementTree as ET
# import numpy as np
# from sklearn.metrics import average_precision_score
# from collections import defaultdict

# # Function to calculate IoU (Intersection over Union)
# def calculate_iou(pred_box, gt_box):
#     x1, y1, x2, y2 = pred_box
#     x1_gt, y1_gt, x2_gt, y2_gt = gt_box

#     # Calculate intersection
#     inter_x1 = max(x1, x1_gt)
#     inter_y1 = max(y1, y1_gt)
#     inter_x2 = min(x2, x2_gt)
#     inter_y2 = min(y2, y2_gt)
    
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#     # Calculate union
#     pred_area = (x2 - x1) * (y2 - y1)
#     gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
#     union_area = pred_area + gt_area - inter_area
    
#     # Calculate IoU
#     iou = inter_area / union_area if union_area > 0 else 0
#     return iou

# # Function to evaluate on the dataset
# def evaluate_model(model, processor, image_dir, annotation_dir, threshold=0.5):
#     # Store the results and ground truths
#     all_detections = []
#     all_ground_truths = []

#     # Iterate over all images in the dataset
#     for image_name in os.listdir(image_dir):
#         if not image_name.endswith(".jpg"):
#             continue
        
#         image_path = os.path.join(image_dir, image_name)
#         annotation_path = os.path.join(annotation_dir, image_name.replace(".jpg", ".xml"))
        
#         # Load the image
#         image = Image.open(image_path)
        
#         # Prepare input for the model
#         inputs = processor(images=image, return_tensors="pt")
        
#         # Perform object detection
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         # Process the detections
#         target_sizes = torch.tensor([image.size[::-1]])  # height, width
#         results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        
#         # Parse the ground truth annotations (PASCAL VOC format)
#         tree = ET.parse(annotation_path)
#         root = tree.getroot()
        
#         gt_boxes = []
#         gt_labels = []
#         for obj in root.findall("object"):
#             bbox = obj.find("bndbox")
#             xmin = int(bbox.find("xmin").text)
#             ymin = int(bbox.find("ymin").text)
#             xmax = int(bbox.find("xmax").text)
#             ymax = int(bbox.find("ymax").text)
#             gt_boxes.append([xmin, ymin, xmax, ymax])
#             gt_labels.append(obj.find("name").text)

#         # Append the predictions and ground truths for evaluation
#         for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#             if score > threshold:
#                 prediction = {
#                     "score": score.item(),
#                     "label": label.item(),
#                     "box": box.tolist()
#                 }
#                 all_detections.append(prediction)

#         for gt_box in gt_boxes:
#             ground_truth = {
#                 "label": "brain_region",  # Adjust if there are specific labels
#                 "box": gt_box
#             }
#             all_ground_truths.append(ground_truth)
    
#     # Calculate mAP (mean Average Precision)
#     # This requires predictions and ground truths to be grouped by their labels
#     labels = defaultdict(list)
#     for pred in all_detections:
#         labels[pred["label"]].append(pred)

#     # Calculate IoU, precision, recall for each object detected
#     iou_threshold = 0.5  # You can change this threshold for IoU

#     true_positive = 0
#     false_positive = 0
#     false_negative = 0

#     for gt in all_ground_truths:
#         gt_box = gt["box"]
#         matched = False
#         for pred in all_detections:
#             pred_box = pred["box"]
#             iou = calculate_iou(pred_box, gt_box)
#             if iou > iou_threshold:
#                 matched = True
#                 true_positive += 1
#                 break
#         if not matched:
#             false_negative += 1

#     # False positives are those predictions that don't match any ground truth
#     false_positive = len(all_detections) - true_positive

#     precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
#     recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

#     # Average Precision calculation
#     ap = average_precision_score([1]*true_positive + [0]*false_positive, 
#                                  [p["score"] for p in all_detections])

#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"Average Precision (AP): {ap:.4f}")
#     return precision, recall, ap

# # Load pre-trained DETR model and processor
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# # Define your image and annotation directories
# image_dir = "path_to_images"
# annotation_dir = "path_to_annotations"

# # Evaluate the model
# evaluate_model(model, processor, image_dir, annotation_dir)
