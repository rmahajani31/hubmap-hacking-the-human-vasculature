from mmdet.evaluation.metrics import CocoMetric
from mmyolo.registry import METRICS
from mmdet.structures.mask import encode_mask_results

import torch
import numpy as np
import cv2
import pickle

@METRICS.register_module()
class HubMapDetCocoMetric(CocoMetric):
    def __init__(self, ann_file='', proposal_nums=(100, 300, 1000), metric='bbox', score_thresh=0.01, save_preds=False, preds_file='val_preds.pkl'):
        super(HubMapDetCocoMetric, self).__init__(ann_file=ann_file, proposal_nums=proposal_nums, metric=metric)
        self.score_thresh = score_thresh
        self.save_preds = save_preds
        self.preds_file = preds_file
        
    
    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            result = dict()
            # print('=========COCO METRIC===========')
            # print(data_sample.keys())
            # print('=========COCO METRIC===========')
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['img_path'] = data_sample['img_path']
            result['scores'] = pred['scores'].cpu().numpy()
            tgt_idx = np.where(result['scores'] > self.score_thresh)
            result['scores'] = result['scores'][tgt_idx]
            result['bboxes'] = pred['bboxes'].cpu().numpy()[tgt_idx]
            result['labels'] = pred['labels'].cpu().numpy()[tgt_idx]
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()[tgt_idx]) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks'][tgt_idx]
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()[tgt_idx]

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))
    
    def compute_metrics(self, results):
        if self.save_preds:
            _, preds = zip(*results)
            with open(self.preds_file, 'wb') as f:
                pickle.dump(preds, f)
        return super().compute_metrics(results)