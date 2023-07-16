from mmseg.models.segmentors import EncoderDecoder, BaseSegmentor
from mmseg.models.utils import resize
from mmseg.models.builder import build_segmentor, SEGMENTORS
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask

import torch
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
from mmengine.structures import PixelData

@SEGMENTORS.register_module()
class CustomEncoderDecoder(EncoderDecoder):
    def predict(self,
                inputs,
                data_samples):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation bef0ore normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        new_data_samples = self.postprocess_result(seg_logits, data_samples)
        print('==============PREDICT===============')
        print(new_data_samples[0].metainfo, new_data_samples[0].seg_logits.data.shape, new_data_samples[0].gt_sem_seg.data.shape, new_data_samples[0].pred_sem_seg.data.shape, new_data_samples[0].final_seg_pred.data.shape)
        print('==============PREDICT===============')
        return new_data_samples

    def postprocess_result(self,
                            seg_logits,
                            data_samples):
            """ Convert results list to `SegDataSample`.
            Args:
                seg_logits (Tensor): The segmentation results, seg_logits from
                    model of each input image.
                data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                    It usually includes information such as `metainfo` and
                    `gt_sem_seg`. Default to None.
            Returns:
                list[:obj:`SegDataSample`]: Segmentation results of the
                input images. Each SegDataSample usually contain:

                - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
                - ``seg_logits``(PixelData): Predicted logits of semantic
                    segmentation before normalization.
            """
            batch_size, C, H, W = seg_logits.shape

            if data_samples is None:
                data_samples = [SegDataSample() for _ in range(batch_size)]
                only_prediction = True
            else:
                only_prediction = False

            for i in range(batch_size):
                if not only_prediction:
                    img_meta = data_samples[i].metainfo
                    # remove padding area
                    if 'img_padding_size' not in img_meta:
                        padding_size = img_meta.get('padding_size', [0] * 4)
                    else:
                        padding_size = img_meta['img_padding_size']
                    padding_left, padding_right, padding_top, padding_bottom =\
                        padding_size
                    # i_seg_logits shape is 1, C, H, W after remove padding
                    i_seg_logits = seg_logits[i:i + 1, :,
                                            padding_top:H - padding_bottom,
                                            padding_left:W - padding_right]

                    flip = img_meta.get('flip', None)
                    if flip:
                        flip_direction = img_meta.get('flip_direction', None)
                        assert flip_direction in ['horizontal', 'vertical']
                        if flip_direction == 'horizontal':
                            i_seg_logits = i_seg_logits.flip(dims=(3, ))
                        else:
                            i_seg_logits = i_seg_logits.flip(dims=(2, ))

                    # resize as original shape
                    # i_seg_logits = resize(
                    #     i_seg_logits,
                    #     size=img_meta['ori_shape'],
                    #     mode='bilinear',
                    #     align_corners=self.align_corners,
                    #     warning=False).squeeze(0)
                    i_seg_logits = i_seg_logits.squeeze(0)
                else:
                    i_seg_logits = seg_logits[i]

                if C > 1:
                    i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
                    i_seg_prob = torch.softmax(i_seg_logits, dim=0)[1]
                    i_seg_prob = i_seg_prob.unsqueeze(dim=0).unsqueeze(dim=0)
                    bbox = torch.tensor(data_samples[i].metainfo['bbox'], device='cuda')
                    bbox = bbox.unsqueeze(dim=0)
                    ori_shape = data_samples[i].metainfo['ori_shape']
                    i_seg_prob, _ = _do_paste_mask(i_seg_prob, bbox, ori_shape[0], ori_shape[1], False)
                    final_seg_pred = i_seg_prob >= 0.5
                else:
                    i_seg_logits = i_seg_logits.sigmoid()
                    i_seg_pred = (i_seg_logits >
                                self.decode_head.threshold).to(i_seg_logits)
                print(f'i_seg_logits shape: {i_seg_logits.shape}')
                data_samples[i].set_data({
                    'seg_logits':
                    PixelData(**{'data': i_seg_logits}),
                    'pred_sem_seg':
                    PixelData(**{'data': i_seg_pred}),
                    'final_seg_pred':
                    PixelData(**{'data': final_seg_pred})
                })

            return data_samples
    # def inference(self, inputs, batch_img_metas):
    #     print('===============================')
    #     print(f'Calling inference!!! {inputs.shape}')
    #     print(batch_img_metas)
    #     print('===============================')
    #     """Inference with slide/whole style.

    #     Args:
    #         inputs (Tensor): The input image of shape (N, 3, H, W).
    #         batch_img_metas (List[dict]): List of image metainfo where each may
    #             also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
    #             'ori_shape', 'pad_shape', and 'padding_size'.
    #             For details on the values of these keys see
    #             `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

    #     Returns:
    #         Tensor: The segmentation results, seg_logits from model of each
    #             input image.
    #     """
    #     assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
    #         f'Only "slide" or "whole" test mode are supported, but got ' \
    #         f'{self.test_cfg["mode"]}.'
    #     ori_shape = batch_img_metas[0]['ori_shape']
    #     assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
    #     if self.test_cfg.mode == 'slide':
    #         seg_logit = self.slide_inference(inputs, batch_img_metas)
    #     else:
    #         seg_logit = self.whole_inference(inputs, batch_img_metas)

    #     print(f'Seg Logit shape: {seg_logit.shape}=======')
    #     seg_prob = torch.softmax(seg_logit, dim=1)[:,1]
    #     seg_prob = seg_prob.unsqueeze(dim=0)
    #     bbox = torch.tensor(batch_img_metas[0]['bbox'], device='cuda')
    #     bbox = bbox.unsqueeze(dim=0)
    #     seg_prob, _ = _do_paste_mask(seg_prob, bbox, ori_shape[0], ori_shape[1], False)
    #     print(seg_prob.shape, bbox.shape, seg_prob.squeeze().max(), seg_prob.squeeze().min(), seg_logit.squeeze()[0].max(), seg_logit.squeeze()[0].min())
    #     return seg_prob