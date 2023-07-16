import mmcv

from mmcv.transforms.base import BaseTransform
# from mmcv.transforms import TRANSFORMS
from mmseg.registry import TRANSFORMS

import os
import torch
from torchvision.ops import roi_align

from .utils import *

@TRANSFORMS.register_module()
class LoadSegMask(BaseTransform):
    def __init__(self):
        super().__init__()
    
    def transform(self, results):
        mask = to_mask(results['segmentation'], results['height'], results['width'])
        results['gt_seg_map'] = mask
        results['seg_fields'] = ['gt_seg_map']
        return results

@TRANSFORMS.register_module()
class BoxJitter(BaseTransform):
    def __init__(self, jittor_range=(0.8, 1.2), prob=0.5):
        super().__init__()
        self.jittor_range = jittor_range
        self.prob = prob
    
    def transform(self, results):
        if np.random.random() >= self.prob:
            return results

        x1, y1, x2, y2 = results['bbox']
        img_h, img_w = results['ori_shape']
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2

        t = (yc - y1) * np.random.uniform(*self.jittor_range)
        l = (xc - x1) * np.random.uniform(*self.jittor_range)
        b = (y2 - yc) * np.random.uniform(*self.jittor_range)
        r = (x2 - xc) * np.random.uniform(*self.jittor_range)

        x1 = xc - l
        y1 = yc - t
        x2 = xc + r
        y2 = yc + b

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        if y2 > y1 and x2 > x1:
            results['bbox'] = [x1, y1, x2, y2]
            return results
        else:
            print('Invalid box:', x1, y1, x2, y2)
            return results


@TRANSFORMS.register_module()
class ROIAlign(BaseTransform):
    def __init__(self, output_size=(128, 128), spatial_scale=1.0, sampling_ratio=0, aligned=True):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
    
    def transform(self, results):
        x1, y1, x2, y2 = results['bbox']  # xyxy
        rois = torch.FloatTensor([[0, x1, y1, x2, y2]])
        
        img = results['img']
        input = torch.from_numpy(img.transpose(2, 0, 1)[None, ...]).float()
        img_crop = roi_align(
            input, rois, self.output_size, self.spatial_scale,
            self.sampling_ratio, self.aligned
        )[0].numpy().transpose(1, 2, 0)

        results['img'] = img_crop
        results['img_shape'] = img_crop.shape[:2]
        results['pad_shape'] = img_crop.shape[:2]
        results['scale_factor'] = 1.0
        results['keep_ratio'] = False

        if 'gt_seg_map' in results:
            mask = results['gt_seg_map']
            input = torch.from_numpy(mask[None, None, ...]).float()
            mask_crop = roi_align(
                input, rois, self.output_size, self.spatial_scale,
                self.sampling_ratio, self.aligned
            )[0, 0].numpy()
            mask_crop = (mask_crop >= 0.5).astype(int)
            results['gt_seg_map'] = mask_crop
        return results

@TRANSFORMS.register_module()
class BBoxToTensor(BaseTransform):
    def __init__(self):
        super().__init__()
    
    def transform(self, results):
        new_bbox = torch.tensor(results['bbox'], device='cuda')
        results['bbox'] = new_bbox
        return results

@TRANSFORMS.register_module()
class FormatSegMask(BaseTransform):
    def __init__(self):
        super().__init__()
    
    def transform(self, results):
        mask_crop = results['gt_seg_map'].copy()
        mask_crop = mask_crop - 1
        mask_crop[mask_crop == -1] = 255
        results['gt_seg_map'] = mask_crop
        return results