![image](https://github.com/user-attachments/assets/6abb2841-e30d-42f8-bcfb-575ffd3fb6d2)

``` 
import torch
import torchvision.transforms as T
from transformers import DetrForObjectDetection, DetrImageProcessor

# Load model and image processor
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

# Define image transformations
transform = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor()
])

# Load and preprocess the image
image_path = 'path/to/your/jn.jpg'
image = T.Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)

# Pass the image through the model
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Get the detected objects
logits = outputs.logits
boxes = outputs.pred_boxes

print("Detected objects:", logits)
print("Bounding boxes:", boxes)
``` 
