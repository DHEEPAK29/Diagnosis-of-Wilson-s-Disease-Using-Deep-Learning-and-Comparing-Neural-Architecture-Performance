import torch
import torchvision.transforms as transforms
from fcos_core.config import cfg
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.structures.image_list import to_image_list
from fcos_core.utils import cv2_util

# Load the configuration file
cfg.merge_from_file("path/to/config/file.yaml")
cfg.freeze()

# Build the model
model = build_detection_model(cfg)
model.eval()

# Load the model weights
checkpointer = DetectronCheckpointer(cfg, model)
checkpointer.load("path/to/model/weights.pth")

# Define the image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
])

# Load and preprocess the image
image = cv2.imread("path/to/brain/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
image_list = image_list.to(cfg.MODEL.DEVICE)

# Perform inference
with torch.no_grad():
    predictions = model(image_list)

# Process the predictions
predictions = [o.to(torch.device("cpu")) for o in predictions]
boxes = predictions[0].bbox
scores = predictions[0].get_field("scores")
labels = predictions[0].get_field("labels")

# Visualize the results
for box, score, label in zip(boxes, scores, labels):
    if score > 0.5:  # Threshold for visualization
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
