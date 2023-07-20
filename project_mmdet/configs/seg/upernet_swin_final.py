_base_ = [
    '/home/ec2-user/hubmap-hacking-the-human-vasculature/mmsegmentation/configs/_base_/models/upernet_swin.py',
    '/home/ec2-user/hubmap-hacking-the-human-vasculature/mmsegmentation/configs/_base_/default_runtime.py', '/home/ec2-user/hubmap-hacking-the-human-vasculature/mmsegmentation/configs/_base_/schedules/schedule_160k.py'
]

classes = ('unlabelled', 'blood_vessel')
num_classes = len(classes)

crop_size = (224, 224)
data_preprocessor = dict(type='CustomSegDataPreProcessor', size=crop_size)
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=500)

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    type='CustomEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
        decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=num_classes, out_channels=1, loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        auxiliary_head=dict(in_channels=384, num_classes=num_classes, out_channels=1, loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

dataset_type = 'HubMapSegTrainDataset'
generate_all_datset_annots = True
base_data_dir_name_1 = 'dataset1_files' if not generate_all_datset_annots else 'all_dataset_files'
base_data_dir_name_2 = 'all_dataset1' if not generate_all_datset_annots else 'all_dataset'
data_root = f'/home/ec2-user/hubmap-hacking-the-human-vasculature/{base_data_dir_name_1}/{base_data_dir_name_2}_mmdet_fold_0/'
suffix_end = 'only_dataset1' if not generate_all_datset_annots else 'dataset1_and_2'
suffix = f'fold_0_run_upernet_swin_{suffix_end}'

chkp_dir = f'/home/ec2-user/hubmap-hacking-the-human-vasculature/project_mmdet/models_{suffix}'
metrics_file_name = f'metrics_{suffix}.txt'
chkp_name = f'model_{suffix}.pth'

# Path of train annotation file
train_ann_file = 'annotations/train_annotations.json'
train_data_prefix = 'train_images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/validation_annotations.json'
val_data_prefix = 'validation_images/'  # Prefix of val image path

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSegMask'),
    dict(type='BoxJitter'),
    dict(type='ROIAlign', output_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ROIAlign', output_size=crop_size),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PackSegInputs', meta_keys=('img_path', 'img_id', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label', 'bbox', 'score', 'category_id'))
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        img_dir=train_data_prefix,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        img_dir=val_data_prefix,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='HubMapSegCocoMetric',
    proposal_nums=(1000, 1, 10),
    ann_file=data_root + val_ann_file,
    metric=['bbox', 'segm'])
test_evaluator = val_evaluator

custom_hooks = [dict(type='ModelCheckpointingHook', interval=1, metrics_file_name=metrics_file_name, chkp_dir=chkp_dir, chkp_name=chkp_name, tgt_metric='coco/segm_mAP', should_record_epoch=False)]
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth'