import torch
import torchvision.transforms as transforms
from fcos_core.config import cfg
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.structures.image_list import to_image_list
from fcos_core.utils import cv2_util
from fcos_core.data.datasets.evaluation import evaluate
import matplotlib.pyplot as plt
import cv2

# Load the configuration file
cfg.merge_from_file("path/to/config/file.yaml")
cfg.freeze()

# Build the model
model = build_detection_model(cfg)
model.eval()

# Load the model weights
checkpointer = DetectronCheckpointer(cfg, model)
'''

LOOP

'''

checkpointer.load("path/to/model/weights.pth")

# Define the image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
])

# Load and preprocess the image
image_path = "path/to/brain/image.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_transformed = transform(image_rgb)
image_list = to_image_list(image_transformed, cfg.DATALOADER.SIZE_DIVISIBILITY)
image_list = image_list.to(cfg.MODEL.DEVICE)

# Perform inference
with torch.no_grad():
    predictions = model(image_list)

# Process the predictions
predictions = [o.to(torch.device("cpu")) for o in predictions]
boxes = predictions[0].bbox
scores = predictions[0].get_field("scores")
labels = predictions[0].get_field("labels")

# Draw bounding boxes and labels on the image
for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box.int().tolist()
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
    cv2.putText(image_rgb, f"Label: {label}, Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Evaluate the model
results = evaluate(
    dataset="path/to/brain/image/dataset",
    predictions=predictions,
    output_folder="path/to/output/folder",
    box_only=False,
    iou_types=("bbox",),
    expected_results=[],
    expected_results_sigma_tol=4,
)

# Print evaluation results
print("Evaluation Results:")
for metric, value in results.items():
    print(f"{metric}: {value}")

# Plot the image with detections
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.title("Detection Results")
plt.show()


# import torch
# import torchvision.transforms as transforms
# from fcos_core.config import cfg
# from fcos_core.modeling.detector import build_detection_model
# from fcos_core.utils.checkpoint import DetectronCheckpointer
# from fcos_core.structures.image_list import to_image_list
# from fcos_core.utils import cv2_util
# from fcos_core.data.datasets.evaluation import evaluate

# # Load the configuration file
# cfg.merge_from_file("path/to/config/file.yaml")
# cfg.freeze()

# # Build the model
# model = build_detection_model(cfg)
# model.eval()

# # Load the model weights
# checkpointer = DetectronCheckpointer(cfg, model)
# checkpointer.load("path/to/model/weights.pth")

# # Define the image transformation
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
# ])

# # Load and preprocess the image
# image = cv2.imread("path/to/brain/image.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = transform(image)
# image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
# image_list = image_list.to(cfg.MODEL.DEVICE)

# # Perform inference
# with torch.no_grad():
#     predictions = model(image_list)

# # Process the predictions
# predictions = [o.to(torch.device("cpu")) for o in predictions]
# boxes = predictions[0].bbox
# scores = predictions[0].get_field("scores")
# labels = predictions[0].get_field("labels")

# # Evaluate the model
# results = evaluate(
#     dataset="path/to/brain/image/dataset",
#     predictions=predictions,
#     output_folder="path/to/output/folder",
#     box_only=False,
#     iou_types=("bbox",),
#     expected_results=[],
#     expected_results_sigma_tol=4,
# )

# # Print evaluation results
# print("Evaluation Results:")
# for metric, value in results.items():
#     print(f"{metric}: {value}")
