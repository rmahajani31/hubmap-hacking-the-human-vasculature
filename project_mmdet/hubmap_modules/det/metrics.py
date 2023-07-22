from mmdet.evaluation.metrics import CocoMetric
from mmengine.registry import METRICS
from mmdet.structures.mask import encode_mask_results

import torch
import numpy as np
import cv2
import pickle
from ensemble_boxes import *

@METRICS.register_module()
class HubMapDetCocoMetric(CocoMetric):
    def __init__(self, ann_file='', proposal_nums=(100, 300, 1000), metric='bbox', score_thresh=0.01, save_preds=False, preds_file='val_preds.pkl', save_gt=False, gts_file='val_gt.pkl', save_suffix='', format_only=False, backend_args=None):
        super(HubMapDetCocoMetric, self).__init__(ann_file=ann_file, proposal_nums=proposal_nums, metric=metric, format_only=format_only, backend_args=backend_args)
        self.score_thresh = score_thresh
        self.save_preds = save_preds
        self.preds_file = preds_file
        self.save_gt = save_gt
        self.gts_file = gts_file
        self.save_suffix = save_suffix
    
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
        gts, preds = zip(*results)
        if self.save_preds:
            pred_file_name = self.preds_file if self.save_suffix=='' else f"{self.preds_file.split('.pkl')[0]}_{self.save_suffix}.pkl"
            with open(pred_file_name, 'wb') as f:
                pickle.dump(preds, f)
        if self.save_gt:
            gt_file_name = self.gts_file if self.save_suffix=='' else f"{self.gts_file.split('.pkl')[0]}_{self.save_suffix}.pkl"
            with open(gt_file_name, 'wb') as f:
                pickle.dump(gts, f)
        return super().compute_metrics(results)

@METRICS.register_module()
class HubMapDetEnsembleCocoMetric(CocoMetric):
    def __init__(self, ann_file='', proposal_nums=(100, 300, 1000), metric='bbox', save_preds=False, preds_file='val_preds.pkl', save_suffix='', input_preds_files=[], input_pred_weights=[]):
        super(HubMapDetEnsembleCocoMetric, self).__init__(ann_file=ann_file, proposal_nums=proposal_nums, metric=metric)
        self.save_preds = save_preds
        self.preds_file = preds_file
        self.save_suffix = save_suffix
        self.input_preds_files = input_preds_files
        self.input_pred_weights = input_pred_weights
        self.img_id_to_results = dict()
        self.dataset_meta = {'classes': ('blood_vessel,')}
    
    def compute_metrics(self, results):
        iou_thr = 0.5
        skip_box_thr = 0.0001
        new_results = []
        tgt_img_id = 0
        for i, input_pred_file in enumerate(self.input_preds_files):
            with open(input_pred_file, 'rb') as f:
                cur_results = pickle.load(f)
            for cur_result in cur_results:
                img_id = cur_result['img_id']
                img_path = cur_result['img_path']
                scores = cur_result['scores'].tolist()
                bboxes = (cur_result['bboxes']/512).tolist()
                # if img_id == tgt_img_id:
                    # print('==========BBOX STATS===========')
                    # print(cur_result['bboxes'].dtype, cur_result['scores'].dtype, cur_result['labels'].dtype, (cur_result['bboxes']/512).min(), (cur_result['bboxes']/512).max())
                    # print('==========BBOX STATS===========')
                labels = cur_result['labels'].tolist()
                if img_id in self.img_id_to_results:
                    self.img_id_to_results[img_id][1].append(bboxes)
                    self.img_id_to_results[img_id][2].append(scores)
                    self.img_id_to_results[img_id][3].append(labels)
                else:
                    self.img_id_to_results[img_id] = (img_path, [bboxes], [scores], [labels])
        # print(len(self.img_id_to_results[tgt_img_id][1]), len(self.img_id_to_results[tgt_img_id][2]), len(self.img_id_to_results[tgt_img_id][3]))
        for img_id in self.img_id_to_results.keys():
            img_path = self.img_id_to_results[img_id][0]
            boxes, scores, labels = weighted_boxes_fusion(self.img_id_to_results[img_id][1], self.img_id_to_results[img_id][2], self.img_id_to_results[img_id][3], weights=self.input_pred_weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            boxes = np.array(boxes, dtype=np.float32) * 512
            scores = np.array(scores, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            new_result = {'img_id': img_id, 'img_path': img_path, 'bboxes': boxes, 'scores': scores, 'labels': labels}
            new_results.append(({'img_id': img_id, 'height': 512, 'width': 512}, new_result))
        gts, preds = zip(*new_results)
        if self.save_preds:
            pred_file_name = self.preds_file if self.save_suffix=='' else f"{self.preds_file.split('.pkl')[0]}_{self.save_suffix}.pkl"
            with open(pred_file_name, 'wb') as f:
                pickle.dump(preds, f)
        return super().compute_metrics(new_results)