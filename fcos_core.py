import torch
import torchvision.transforms as transforms
from fcos_core.config import cfg
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.structures.image_list import to_image_list
from fcos_core.utils import cv2_util
from matplotlib import pyplot as plt

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

# Define the training loop with loss and accuracy tracking
def train(net, train_data, val_data, epochs, ctx):
    trainer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for i, (images, targets) in enumerate(train_data):
            images = images.to(ctx)
            targets = targets.to(ctx)
            trainer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, targets)
            loss.backward()
            trainer.step()
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_loss.append(epoch_loss / len(train_data))
        train_accuracy.append(100. * correct / total)
        print(f'Epoch {epoch} completed with training loss: {train_loss[-1]}, accuracy: {train_accuracy[-1]}%')

        net.eval()
        val_epoch_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_data):
                images = images.to(ctx)
                targets = targets.to(ctx)
                outputs = net(images)
                loss = criterion(outputs, targets)
                val_epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_loss.append(val_epoch_loss / len(val_data))
        val_accuracy.append(100. * correct / total)
        print(f'Epoch {epoch} completed with validation loss: {val_loss[-1]}, accuracy: {val_accuracy[-1]}%')

    # Plotting the loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.show()

# Load data
train_data = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Train the model
ctx = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, train_data, val_data, epochs=10, ctx=ctx)
