import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load pre-trained DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Load the image
img_path = "path_to_brain_image.jpg"
image = Image.open(img_path)

# Prepare the image for DETR
inputs = processor(images=image, return_tensors="pt")

def save_pascal_voc_annotation(image_path, objects, output_xml_path):
    tree = ET.ElementTree(ET.Element("annotation"))
    root = tree.getroot()
    
    # Add image size
    filename = os.path.basename(image_path)
    img = Image.open(image_path)
    width, height = img.size
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"  # Assuming RGB image
    
    # Add each detected object
    for obj in objects:
        obj_elem = ET.SubElement(root, "object")
        ET.SubElement(obj_elem, "name").text = obj["label"]
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"
        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(obj["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(obj["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(obj["ymax"])
    
    tree.write(output_xml_path)

# Example of how to use the function after detection:
objects = []
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.9:  # Set a threshold for detection confidence
        objects.append({
            "label": f"brain_region_{label.item()}",  # Replace with specific region name if needed
            "xmin": round(box[0].item(), 2),
            "ymin": round(box[1].item(), 2),
            "xmax": round(box[2].item(), 2),
            "ymax": round(box[3].item(), 2)
        })

save_pascal_voc_annotation(img_path, objects, "output_annotation.xml")

# Perform object detection
with torch.no_grad():
    outputs = model(**inputs)

# Extract bounding boxes and labels
target_sizes = torch.tensor([image.size[::-1]])  # height, width
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# Plot the results
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

# Add the bounding boxes
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)

# Show the image with detections
plt.show()

# Optionally, save the output image with bounding boxes
image_with_boxes = image.copy()
plt.figure(figsize=(12, 9))
plt.imshow(image_with_boxes)
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    plt.gca().add_patch(rect)
plt.savefig("brain_image_with_boxes.jpg")


# <annotation>
#     <folder>images</folder>
#     <filename>brain_image.jpg</filename>
#     <size>
#         <width>640</width>
#         <height>480</height>
#         <depth>3</depth>
#     </size>
#     <object>
#         <name>brain_region_1</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>50</xmin>
#             <ymin>50</ymin>
#             <xmax>200</xmax>
#             <ymax>200</ymax>
#         </bndbox>
#     </object>
#     <!-- Add more objects as needed -->
# </annotation>

