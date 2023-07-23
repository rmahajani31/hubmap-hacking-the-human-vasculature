from mmseg.models.segmentors import EncoderDecoder, BaseSegmentor
from mmseg.models.utils import resize
from mmseg.models.builder import build_segmentor, SEGMENTORS
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask

from mmengine.config import Config, ConfigDict, DictAction
 
import torch
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
from mmengine.structures import PixelData
import torch.nn as nn
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from typing import List, Optional

from mmseg.apis import init_model
import os

@SEGMENTORS.register_module()
class EnsembleSegmentor(BaseSegmentor):
    def __init__(
        self,
        configs,
        checkpoints,
        data_preprocessor,
        weights,
        train_cfg=None,
        test_cfg=None,
        **kwargs
    ):
        super(EnsembleSegmentor, self).__init__(data_preprocessor=data_preprocessor)
        self.models = nn.ModuleList()
        for config, checkpoint in zip(configs, checkpoints):
            model = init_model(os.path.abspath(config), checkpoint)
            self.models.append(model)
        self.weights = weights

        self.test_cfg = test_cfg

    def ensemble_inference(self, inputs, batch_img_metas):
        seg_logits = []
        for model, weight in zip(self.models, self.weights):
            seg_logit = model.inference(inputs, batch_img_metas)
            seg_logits.append(seg_logit * weight)
        seg_logit = sum(seg_logits) / sum(self.weights)
        return seg_logit

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
        

        seg_logits = self.ensemble_inference(inputs, batch_img_metas)

        new_data_samples = self.postprocess_result(seg_logits, data_samples)
        # print('==============PREDICT===============')
        # # print(new_data_samples[0].metainfo, new_data_samples[0].seg_logits.data.shape, new_data_samples[0].pred_sem_seg.data.shape, new_data_samples[0].final_seg_pred.data.shape)
        # print(len(new_data_samples))
        # print('==============PREDICT===============')
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
                    # print('========IN POSTPROC MULTIPLE CHANNELS===============')
                    # print(i_seg_logits.shape, self.decode_head.threshold)
                    # print('========IN POSTPROC MULTIPLE CHANNELS===============')
                    i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
                    i_seg_prob = torch.softmax(i_seg_logits, dim=0)[1]
                    i_seg_prob = i_seg_prob.unsqueeze(dim=0).unsqueeze(dim=0)
                    bbox = torch.tensor(data_samples[i].metainfo['bbox'], device='cuda')
                    bbox = bbox.unsqueeze(dim=0)
                    ori_shape = data_samples[i].metainfo['ori_shape']
                    i_seg_prob, _ = _do_paste_mask(i_seg_prob, bbox, ori_shape[0], ori_shape[1], False)
                    final_seg_pred = i_seg_prob >= 0.5
                else:
                    # print('========IN POSTPROC SINGLE CHANNELS===============')
                    # print(i_seg_logits.shape, self.decode_head.threshold)
                    # print('========IN POSTPROC SINGLE CHANNELS===============')
                    i_seg_logits = i_seg_logits.sigmoid()
                    i_seg_prob = i_seg_logits.unsqueeze(dim=0)
                    bbox = torch.tensor(data_samples[i].metainfo['bbox'], device='cuda')
                    bbox = bbox.unsqueeze(dim=0)
                    ori_shape = data_samples[i].metainfo['ori_shape']
                    i_seg_prob, _ = _do_paste_mask(i_seg_prob, bbox, ori_shape[0], ori_shape[1], False)
                    final_seg_pred = i_seg_prob > 0.3
                    i_seg_pred = (i_seg_logits >
                                0.3).to(i_seg_logits)
                # print(f'i_seg_logits shape: {i_seg_logits.shape}')
                data_samples[i].set_data({
                    'seg_logits':
                    PixelData(**{'data': i_seg_logits}),
                    'pred_sem_seg':
                    PixelData(**{'data': i_seg_pred}),
                    'final_seg_pred':
                    PixelData(**{'data': final_seg_pred})
                })

            return data_samples
    

    def loss(self, inputs, data_samples):
        """Calculate losses from a batch of inputs and data samples."""
        pass

    def _forward(self,
                 inputs,
                 data_samples):
        """Network forward process.
        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def encode_decode(self, inputs, batch_data_samples):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    def extract_feat(self, inputs):
        """Placeholder for extract features from images."""
        pass