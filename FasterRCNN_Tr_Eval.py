import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import matplotlib.pyplot as plt

# Load a pre-trained ResNet50 model
backbone = torchvision.models.resnet50(pretrained=True)
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
backbone.out_channels = 2048

# Define the RPN anchor generator
rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

# Define the ROI align layer
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2
)

# Define the Cascade R-CNN model
model = FasterRCNN(
    backbone,
    num_classes=2,  # 1 class (brain) + background
    rpn_anchor_generator=rpn_anchor_generator,
    box_roi_pool=roi_pooler
)

# Define the transform
transform = GeneralizedRCNNTransform(
    min_size=800,
    max_size=1333,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225]
)

# Load dataset 
dataset = Dataset(root='/dir/brain+ve', transforms=transform)

# Split the dataset into training and validation sets
indices = torch.randperm(len(dataset)).tolist()
train_dataset = torch.utils.data.Subset(dataset, indices[:-50])
val_dataset = torch.utils.data.Subset(dataset, indices[-50:])

# Define the data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=lambda x: tuple(zip(*x))
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=4, shuffle=False, num_workers=4,
    collate_fn=lambda x: tuple(zip(*x))
)

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 13
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            val_losses = sum(loss for loss in loss_dict.values())
    
    print(f"Epoch {epoch+1}, Training Loss: {losses.item()}, Validation Loss: {val_losses.item()}")

print("Training complete!")
