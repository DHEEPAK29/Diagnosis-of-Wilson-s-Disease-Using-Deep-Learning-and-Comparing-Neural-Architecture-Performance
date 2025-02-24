import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class MyDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)
        annotation_path = os.path.join(self.annotation_dir, image_name.split('.')[0] + '.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            image = self.transforms(image)

        return {
            'image': torch.tensor(image).permute(2, 0, 1).float(),
            'annotation': torch.tensor(annotation).long()
        }


# Load SAM2 Meta Model
sam_checkpoint = "_CHECKPOINT"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
mask_generator = SamAutomaticMaskGenerator(sam)


# Hyperparameters
image_dir = 'PATH'
annotation_dir = 'PATH'
batch_size = 4
num_epochs = 10
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset and create data loaders
dataset = MyDataset(image_dir, annotation_dir, transforms=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer
optimizer = optim.Adam(sam.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        images = batch['image'].to(device)
        annotations = batch['annotation'].to(device)

        optimizer.zero_grad()
        
        # Generate masks using SAM2
        outputs = mask_generator.generate(images)
        
        # Convert output masks to tensor
        outputs_tensor = torch.tensor(np.array([output["segmentation"] for output in outputs])).to(device)
        
        loss = loss_fn(outputs_tensor, annotations)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save model checkpoint
torch.save(sam.state_dict(), 'sam2_meta_model.pth')


# Make predictions
def make_predictions(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    return masks


# Load trained model
sam.load_state_dict(torch.load('sam2_meta_model.pth'))
image_path = 'path/to/image.jpg'
prediction = make_predictions(sam, image_path)

# Visualize prediction
plt.imshow(prediction[0]['segmentation'], cmap='gray')
plt.show()
