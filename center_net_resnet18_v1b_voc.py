import mxnet as mx
from mxnet import autograd, gluon
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from gluoncv import data, model_zoo, utils
from matplotlib import pyplot as plt
import numpy as np

# Load the Pascal VOC dataset
train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = data.VOCDetection(splits=[(2007, 'test')])

print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))

# Load a pre-trained CenterNet model
net = model_zoo.get_model('center_net_resnet18_v1b_voc', pretrained=True)

# Pre-process an example image
im_fname = utils.download('PATH', path='sampl1_X1AEVO.jpg')
x, img = data.transforms.presets.center_net.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)

# Define the training loop
def train(net, train_data, val_data, epochs, ctx):
    trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001, 'momentum': 0.9, 'wd': 0.0005})
    for epoch in range(epochs):
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [net.loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            trainer.step(batch_size)
        print(f'Epoch {epoch} completed')

# Load data
train_data = DataLoader(train_dataset.transform(data.transforms.presets.center_net.CenterNetDefaultTrainTransform(512, 512)), batch_size=8, shuffle=True)
val_data = DataLoader(val_dataset.transform(data.transforms.presets.center_net.CenterNetDefaultValTransform(512, 512)), batch_size=8, shuffle=False)

# Train the model
ctx = [mx.gpu(0)]
train(net, train_data, val_data, epochs=10, ctx=ctx)

# Evaluate the model
for i, batch in enumerate(val_data):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    outputs = [net(X) for X in data]
