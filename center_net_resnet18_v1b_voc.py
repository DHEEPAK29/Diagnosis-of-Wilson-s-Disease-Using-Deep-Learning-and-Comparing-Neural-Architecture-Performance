import mxnet as mx
from mxnet import autograd, gluon
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from gluoncv import data, model_zoo, utils
from matplotlib import pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

transform = transforms.Compose([
    transforms.ToTensor()
])

# Pascal VOC dataset
train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = data.VOCDetection(splits=[(2007, 'test')])

print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))

image, target = train_dataset[0]

plt.imshow(image.permute(1, 2, 0))  
plt.axis('off')
plt.show()

# Load a pre-trained CenterNet model
net = model_zoo.get_model('center_net_resnet18_v1b_voc', pretrained=True)

# Pre-process an example image
im_fname = utils.download('PATH', path='sampl1_X1AEVO.jpg')
x, img = data.transforms.presets.center_net.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

# Define the training loop
def train(net, train_data, val_data, epochs, ctx):
    trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001, 'momentum': 0.9, 'wd': 0.0005})
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [net.loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            trainer.step(batch_size)
            epoch_loss += sum([l.mean().asscalar() for l in losses]) / len(losses)
        train_loss.append(epoch_loss / len(train_data))
        print(f'Epoch {epoch} completed with training loss: {train_loss[-1]}')

        # Validation loss
        val_epoch_loss = 0
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            losses = [net.loss(yhat, y) for yhat, y in zip(outputs, label)]
            val_epoch_loss += sum([l.mean().asscalar() for l in losses]) / len(losses)
        val_loss.append(val_epoch_loss / len(val_data))
        print(f'Epoch {epoch} completed with validation loss: {val_loss[-1]}')

    # Loss plot 
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

# Load data
train_data = DataLoader(train_dataset.transform(data.transforms.presets.center_net.CenterNetDefaultTrainTransform(512, 512)), batch_size=8, shuffle=True)
val_data = DataLoader(val_dataset.transform(data.transforms.presets.center_net.CenterNetDefaultValTransform(512, 512)), batch_size=8, shuffle=False)

# Training the model
ctx = [mx.gpu(0)]
train(net, train_data, val_data, epochs=10, ctx=ctx)
