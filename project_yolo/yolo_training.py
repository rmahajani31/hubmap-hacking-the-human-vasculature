from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_folds = 1
import gc

for i in range(num_folds):
    model = YOLO('yolov8m-seg.pt')
    # custom_config = {
    # 'hsv_h': 0.015,
    # 'hsv_s': 0.7,
    # 'hsv_v': 0.4,
    # 'degrees': 0.0,
    # 'translate': 0.1,
    # 'scale': 0.9,
    # 'shear': 0.0,
    # 'perspective': 0.0,
    # 'flipud': 0.0,
    # 'fliplr': 0.5,
    # 'mosaic': 1.0,
    # 'mixup': 0.1,
    # 'copy_paste': 0.1}

    model_config = {'data': f'hubmap_{i}.yaml', 'epochs': 100, 'imgsz': 640, 'save': True, 'save_period': 100, 'plots': True, 'device': device}
    # 'mosaic': 0, 'mixup': 0, 'translate': 0, 'scale': 0
    # model_config.update(custom_config)
    model.train(**model_config)
    gc.collect()
gc.collect()




