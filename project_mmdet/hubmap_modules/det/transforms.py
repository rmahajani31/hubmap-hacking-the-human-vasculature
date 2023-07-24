from mmyolo.registry import TRANSFORMS
from mmyolo.datasets.transforms import LoadAnnotations as MMYOLO_LoadAnnotations
from mmdet.datasets.transforms import PackDetInputs as MMDET_PackDetInputs

import numpy as np

@TRANSFORMS.register_module()
class LoadAnnotationsHubMap(MMYOLO_LoadAnnotations):
    def transform(self, results):
        new_results = super().transform(results)
        # print('=========RESULTS BEFORE PROC SCORES=======')
        # print(results)
        # print('=========RESULTS BEFORE PROC SCORES=======')
        if len(new_results.get('instances', [])) != new_results['gt_bboxes'].shape[0]:
            print('========LEN OF results in SUBCLASS=====')
            print(new_results)
            print('========LEN OF results in SUBCLASS=====')
        gt_scores = []
        for instance in new_results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_scores.append(instance['bbox_score'])
        new_results['gt_scores'] = np.array(gt_scores, dtype=np.float32)
        # if results['img_path'] == '/home/ec2-user/hubmap-hacking-the-human-vasculature/all_dataset_files/all_dataset_pseudo_labels_mmdet_fold_0/train_images/4d11f2560b3d.tif':
        # print('========AFTER=====')
        # print(results)
        # print('========AFTER=====')
        return new_results

@TRANSFORMS.register_module()
class PackDetInputsHubMap(MMDET_PackDetInputs):
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_scores': 'scores', 
        'gt_masks': 'masks'
    }
    def transform(self, results):
        # if len(results.get('instances', [])) != results['gt_bboxes'].shape[0]:
        #     print('========LEN OF results in PackDetInputsHubMap=====')
        #     print(results)
        #     print('========LEN OF results in PackDetInputsHubMap=====')
        packed_results = super().transform(results)
        # print('=========PAKCED RESULTS======')
        # print(packed_results)
        # print('=========PAKCED RESULTS======')
        return packed_results