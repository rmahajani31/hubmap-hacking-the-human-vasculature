_base_ = [
    '/home/ec2-user/hubmap-hacking-the-human-vasculature/mmsegmentation/configs/_base_/default_runtime.py', '/home/ec2-user/hubmap-hacking-the-human-vasculature/mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
classes = ('unlabelled', 'blood_vessel')
train_cfg = dict(val_interval=500)
input_size = (224, 224)
test_cfg = dict(size=input_size)
data_preprocessor = dict(
    type='CustomSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=input_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='CustomEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=len(classes),
        out_channels=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=len(classes),
        out_channels=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

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
    dict(type='ROIAlign', output_size=input_size),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ROIAlign', output_size=input_size),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PackSegInputs', meta_keys=('img_path', 'img_id', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label', 'bbox', 'score', 'category_id'))
]

# test_pipeline = [
#     dict(
#         type='MultiScaleFlipAug',
#         scales=input_size,
#         transforms=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadSegMask'),
#             dict(type='ROIAlign', output_size=input_size),
#             dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='PackSegInputs')
#         ]
#     )
# ]

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

# val_dataloader = dict(
#     batch_size=64,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type='HubMapSegTestDataset',
#         preds_file='val_preds.pkl',
#         pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='HubMapSegCocoMetric',
    proposal_nums=(1000, 1, 10),
    ann_file=data_root + val_ann_file,
    metric=['bbox', 'segm'])
test_evaluator = val_evaluator

custom_hooks = [dict(type='ModelCheckpointingHook', interval=1, metrics_file_name=metrics_file_name, chkp_dir=chkp_dir, chkp_name=chkp_name, tgt_metric='coco/segm_mAP', should_record_epoch=False)]
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210526_192340-593b0e13.pth'