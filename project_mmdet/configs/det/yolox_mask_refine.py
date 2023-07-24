_base_ = './yolol_mask_refine.py'

# This config use refining bbox and `YOLOv5CopyPaste`.
# Refining bbox means refining bbox by mask while loading annotations and
# transforming after `YOLOv5RandomAffine`

deepen_factor = 1.00
widen_factor = 1.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120411-079ca8d1.pth'
