import mxnet as mx
from gluoncv import data, model_zoo, utils
from mxnet.gluon.data import DataLoader
from mxnet import nd
import matplotlib.pyplot as plt

# Pascal VOC dataset
val_dataset = data.VOCDetection(splits=[(2007, 'test')])

# pre-trained CenterNet model
net = model_zoo.get_model('center_net_resnet18_v1b_voc', pretrained=True)
net.hybridize()

# data
val_data = DataLoader(val_dataset.transform(data.transforms.presets.center_net.CenterNetDefaultValTransform(512, 512)), batch_size=8, shuffle=False)

#  evaluation function
def evaluate(net, val_data, ctx):
    metric = mx.metric.VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            det_bboxes.append(bboxes.clip(0, x.shape[2]))
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6))
        metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return metric.get()

def plot_detections(img, bboxes, ids, scores, class_names):
    ax = utils.viz.plot_bbox(img, bboxes, scores, ids, class_names=class_names)
    plt.show()

ctx = [mx.gpu(0)]
map_name, mean_ap = evaluate(net, val_data, ctx)
print(f'{map_name}: {mean_ap}')

for i, batch in enumerate(val_data):
    if i >= 3:  # number of examples to plot
        break
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    for x in data:
        ids, scores, bboxes = net(x)
        plot_detections(x[0].asnumpy(), bboxes[0].asnumpy(), ids[0].asnumpy(), scores[0].asnumpy(), val_dataset.classes)

# import mxnet as mx
# from gluoncv import data, model_zoo, utils
# from mxnet.gluon.data import DataLoader
# from mxnet import nd

# # Load the Pascal VOC dataset
# val_dataset = data.VOCDetection(splits=[(2007, 'test')])

# # Load a pre-trained CenterNet model
# net = model_zoo.get_model('center_net_resnet18_v1b_voc', pretrained=True)
# net.hybridize()

# # Load data
# val_data = DataLoader(val_dataset.transform(data.transforms.presets.center_net.CenterNetDefaultValTransform(512, 512)), batch_size=8, shuffle=False)

# # Define the evaluation function
# def evaluate(net, val_data, ctx):
#     metric = mx.metric.VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
#     for batch in val_data:
#         data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
#         label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
#         det_bboxes = []
#         det_ids = []
#         det_scores = []
#         gt_bboxes = []
#         gt_ids = []
#         gt_difficults = []
#         for x, y in zip(data, label):
#             ids, scores, bboxes = net(x)
#             det_ids.append(ids)
#             det_scores.append(scores)
#             det_bboxes.append(bboxes.clip(0, x.shape[2]))
#             gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
#             gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
#             gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6))
#         metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
#     return metric.get()

# # Evaluate the model
# ctx = [mx.gpu(0)]
# map_name, mean_ap = evaluate(net, val_data, ctx)
# print(f'{map_name}: {mean_ap}')
