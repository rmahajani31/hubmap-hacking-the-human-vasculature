from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = YOLO('yolov8s-seg.pt')
# model_config = {'data': 'hubmap.yaml', 'epochs': 100, 'imgsz': 512, 'save': True, 'save_period': 100, 'device': device, 'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0, 'translate': 0.0, 'scale': 0.0, 'flipud': 0.5, 'fliplr': 0.5, 'mosaic': 0.0, 'degrees': 0.0}

custom_config = {
 'hsv_h': 0.015,
 'hsv_s': 0.7,
 'hsv_v': 0.4,
 'degrees': 0.0,
 'translate': 0.1,
 'scale': 0.9,
 'shear': 0.0,
 'perspective': 0.0,
 'flipud': 0.0,
 'fliplr': 0.5,
 'mosaic': 1.0,
 'mixup': 0.1,
 'copy_paste': 0.1}

model_config = {'data': 'hubmap_multi_class.yaml', 'epochs': 25, 'imgsz': 512, 'save': True, 'save_period': 20, 'plots': True, 'device': device}
model_config.update(custom_config)
model.train(**model_config)




