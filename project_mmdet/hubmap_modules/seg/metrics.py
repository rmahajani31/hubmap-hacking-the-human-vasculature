from mmdet.evaluation.metrics import CocoMetric
from mmseg.registry import METRICS
from mmdet.structures.mask import encode_mask_results

import torch
import numpy as np
import cv2
import pickle

@METRICS.register_module()
class HubMapSegCocoMetric(CocoMetric):
    def __init__(self, ann_file='', proposal_nums=(100, 300, 1000), metric='bbox', dilate_mask=False, save_preds=False, preds_file='val_segm_preds.pkl'):
        super(HubMapSegCocoMetric, self).__init__(ann_file=ann_file, proposal_nums=proposal_nums, metric=metric)
        self.dilate_mask = dilate_mask
        self.save_preds = save_preds
        self.preds_file = preds_file
        self.img_id_to_results_idx_dict = dict()

    def process(self, data_batch, data_samples):
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        if len(self.results) == 0:
            self.img_id_to_results_idx_dict = dict()
        for data_sample in data_samples:
            # print('=========COCO METRIC===========')
            # print(data_sample.keys())
            cur_img_id = data_sample['img_id']
            cur_bbox = np.expand_dims(np.array(data_sample['bbox']), axis=0)
            cur_score = np.array([data_sample['score']])
            cur_label = np.array([data_sample['category_id']])
            # print(data_sample['final_seg_pred'].keys())
            cur_mask = data_sample['final_seg_pred']['data'].squeeze().detach().cpu().numpy()
            if self.dilate_mask:
                kernel = np.ones((3, 3), np.uint8)
                dilated_mask = cv2.dilate(cur_mask.astype(np.uint8), kernel, iterations=1)
                cur_mask = dilated_mask.astype(bool)
            # print(f'Cur mask shape: {cur_mask.shape}')
            cur_mask = encode_mask_results([cur_mask])
            # print(cur_img_id, cur_bbox, cur_score, cur_label, cur_mask)
            # print('=========COCO METRIC===========')
            if cur_img_id in self.img_id_to_results_idx_dict:
                result_idx = self.img_id_to_results_idx_dict[cur_img_id]
                self.results[result_idx][1]['bboxes'] = np.concatenate((self.results[result_idx][1]['bboxes'], cur_bbox), axis=0)
                # print(self.results[result_idx][1]['scores'].shape, cur_score.shape)
                self.results[result_idx][1]['scores'] = np.concatenate((self.results[result_idx][1]['scores'], cur_score), axis=0)
                self.results[result_idx][1]['labels'] = np.concatenate((self.results[result_idx][1]['labels'], cur_label), axis=0)
                self.results[result_idx][1]['masks'] += cur_mask
                # print('==============CONCAT RESULTS============')
                # print(self.results[result_idx][1]['img_id'], self.results[result_idx][1]['bboxes'].shape, self.results[result_idx][1]['scores'].shape, self.results[result_idx][1]['labels'].shape, len(self.results[result_idx][1]['masks']))
                # print('==============CONCAT RESULTS============')
            else:
                result = dict()
                result['img_id'] = cur_img_id
                result['bboxes'] = cur_bbox
                result['scores'] = cur_score
                result['labels'] = cur_label
                result['masks'] = cur_mask
                # print('==============RESULTS============')
                # print(result['img_id'], result['bboxes'].shape, result['scores'].shape, result['labels'].shape, len(result['masks']))
                # print('==============RESULTS============')

                # parse gt
                gt = dict()
                gt['width'] = data_sample['ori_shape'][1]
                gt['height'] = data_sample['ori_shape'][0]
                # gt['img_id'] = data_sample['img_id']
                if self._coco_api is None:
                    # TODO: Need to refactor to support LoadAnnotations
                    assert 'instances' in data_sample, \
                        'ground truth is required for evaluation when ' \
                        '`ann_file` is not provided'
                    gt['anns'] = data_sample['instances']
                # add converted result to the results list
                # print('===========HUBMAP COCO METRIC GT===========')
                # print(gt)
                # print('===========HUBMAP COCO METRIC GT===========')
                self.results.append((gt, result))
                self.img_id_to_results_idx_dict[cur_img_id] = len(self.results)-1
    
    def compute_metrics(self, results):
        if self.save_preds:
            _, preds = zip(*results)
            with open(self.preds_file, 'wb') as f:
                pickle.dump(preds, f)
        return super().compute_metrics(results)


