from mmdet.evaluation.metrics import CocoMetric
from mmseg.registry import METRICS

@METRICS.register_module()
class HubMapSegCocoMetric(CocoMetric):
    def process(self, data_batch, data_samples):
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            print('=========COCO METRIC===========')
            print(data_sample.keys())
            print('=========COCO METRIC===========')
            # pred = data_sample['pred_instances']
            # result['img_id'] = data_sample['img_id']
            # result['bboxes'] = pred['bboxes'].cpu().numpy()
            # result['scores'] = pred['scores'].cpu().numpy()
            # result['labels'] = pred['labels'].cpu().numpy()
            # # encode mask to RLE
            # if 'masks' in pred:
            #     result['masks'] = encode_mask_results(
            #         pred['masks'].detach().cpu().numpy()) if isinstance(
            #             pred['masks'], torch.Tensor) else pred['masks']
            # # some detectors use different scores for bbox and mask
            # if 'mask_scores' in pred:
            #     result['mask_scores'] = pred['mask_scores'].cpu().numpy()

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
            print('===========HUBMAP COCO METRIC GT===========')
            print(gt)
            print('===========HUBMAP COCO METRIC GT===========')
            self.results.append((gt, result))