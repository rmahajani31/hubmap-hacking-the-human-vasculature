import os
import cv2
import numpy as np
import pickle
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

__all__ = [
    "HubMapDataset"
]

class HubMapDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'all_dataset1_imgs')
        self.annotation_dir = os.path.join(data_dir, 'all_dataset1_annotations')
        self.image_file_list = os.listdir(self.image_dir)
        self.annotation_file_list = os.listdir(self.annotation_dir)
        
    def __len__(self):
        return len(self.image_file_list)
        
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_file_list[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_file_list[idx])
        
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        record = {
            'file_name': image_path,
            'image_id': idx,
            'height': height,
            'width': width,
        }
        
        with open(annotation_path, 'rb') as f:
            orig_annots = pickle.load(f)
        
        objs = []
        for orig_annot in orig_annots:
            orig_annot['bbox_mode'] = BoxMode.XYWH_ABS
            objs.append(orig_annot)
            
        record['annotations'] = objs
        
        return record
