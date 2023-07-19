import torch
from mmseg.registry import MODELS
from mmseg.utils import stack_batch
from mmseg.models import SegDataPreProcessor

@MODELS.register_module()
class CustomSegDataPreProcessor(SegDataPreProcessor):
    def forward(self, data, training):
        data = self.cast_data(data)  # type: ignore
        # print('=============')
        # print(data)
        # print('=============')
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if type(inputs[0]) is tuple:
            inputs = list(inputs[0])
        # print('============')
        # print(inputs)
        # print(len(inputs), type(inputs[0]), len(inputs[0]))
        # print('============')
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            # assert len(inputs) == 1, (
            #     'Batch inference is not support currently, '
            #     'as the image size might be different in a batch')
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    # print('=======PAD INFO=============')
                    # print(pad_info)
                    # print('============================')
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)
