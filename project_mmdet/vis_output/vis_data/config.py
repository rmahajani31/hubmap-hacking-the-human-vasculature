norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(
        224,
        224,
    ))
model = dict(
    type='CustomEncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(
            224,
            224,
        )),
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[
            2,
            2,
            6,
            2,
        ],
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        strides=(
            4,
            2,
            2,
            2,
        ),
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
        )),
    decode_head=dict(
        type='UPerHead',
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        out_channels=1),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4),
        out_channels=1),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer',
    save_dir='vis_output')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = 'models_fold_0_run_upernet_swin_dataset1_and_2_partial_data_final/model_fold_0_run_upernet_swin_dataset1_and_2_0.347.pth'
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-05, betas=(
            0.9,
            0.999,
        ), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True))
crop_size = (
    224,
    224,
)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
classes = (
    'unlabelled',
    'blood_vessel',
)
num_classes = 2
dataset_type = 'HubMapSegTrainDataset'
generate_all_datset_annots = True
base_data_dir_name_1 = 'all_dataset_files'
base_data_dir_name_2 = 'all_dataset'
data_root = '/home/ec2-user/hubmap-hacking-the-human-vasculature/all_dataset_files/all_dataset_mmdet_fold_0/'
suffix_end = 'dataset1_and_2'
suffix = 'fold_0_run_upernet_swin_dataset1_and_2'
chkp_dir = '/home/ec2-user/hubmap-hacking-the-human-vasculature/project_mmdet/models_fold_0_run_upernet_swin_dataset1_and_2'
metrics_file_name = 'metrics_fold_0_run_upernet_swin_dataset1_and_2.txt'
chkp_name = 'model_fold_0_run_upernet_swin_dataset1_and_2.pth'
train_ann_file = 'annotations/train_annotations.json'
train_data_prefix = 'train_images/'
val_ann_file = 'annotations/validation_annotations.json'
val_data_prefix = 'validation_images/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSegMask'),
    dict(type='BoxJitter'),
    dict(type='ROIAlign', output_size=(
        224,
        224,
    )),
    dict(type='RandomFlip', prob=0.5, direction=[
        'horizontal',
        'vertical',
    ]),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ROIAlign', output_size=(
        224,
        224,
    )),
    dict(type='RandomFlip', prob=0.5, direction=[
        'horizontal',
        'vertical',
    ]),
    dict(
        type='PackSegInputs',
        meta_keys=(
            'img_path',
            'img_id',
            'seg_map_path',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'reduce_zero_label',
            'bbox',
            'score',
            'category_id',
        )),
]
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='HubMapSegTrainDataset',
        data_root=
        '/home/ec2-user/hubmap-hacking-the-human-vasculature/all_dataset_files/all_dataset_mmdet_fold_0/',
        ann_file='annotations/train_annotations.json',
        img_dir='train_images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadSegMask'),
            dict(type='BoxJitter'),
            dict(type='ROIAlign', output_size=(
                224,
                224,
            )),
            dict(
                type='RandomFlip',
                prob=0.5,
                direction=[
                    'horizontal',
                    'vertical',
                ]),
            dict(type='PackSegInputs'),
        ]))
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='HubMapSegTestDataset',
        preds_file='val_preds.pkl',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ROIAlign', output_size=(
                224,
                224,
            )),
            dict(
                type='RandomFlip',
                prob=0.5,
                direction=[
                    'horizontal',
                    'vertical',
                ]),
            dict(
                type='PackSegInputs',
                meta_keys=(
                    'img_path',
                    'img_id',
                    'seg_map_path',
                    'ori_shape',
                    'img_shape',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'reduce_zero_label',
                    'bbox',
                    'score',
                    'category_id',
                )),
        ]))
test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='HubMapSegTestDataset',
        preds_file='val_preds.pkl',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ROIAlign', output_size=(
                224,
                224,
            )),
            dict(
                type='RandomFlip',
                prob=0.5,
                direction=[
                    'horizontal',
                    'vertical',
                ]),
            dict(
                type='PackSegInputs',
                meta_keys=(
                    'img_path',
                    'img_id',
                    'seg_map_path',
                    'ori_shape',
                    'img_shape',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'reduce_zero_label',
                    'bbox',
                    'score',
                    'category_id',
                )),
        ]))
val_evaluator = dict(
    type='HubMapSegCocoMetric',
    proposal_nums=(
        1000,
        1,
        10,
    ),
    ann_file=
    '/home/ec2-user/hubmap-hacking-the-human-vasculature/all_dataset_files/all_dataset_mmdet_fold_0/annotations/validation_annotations.json',
    metric=[
        'bbox',
        'segm',
    ],
    dilate_mask=False)
test_evaluator = dict(
    type='HubMapSegCocoMetric',
    proposal_nums=(
        1000,
        1,
        10,
    ),
    ann_file=
    '/home/ec2-user/hubmap-hacking-the-human-vasculature/all_dataset_files/all_dataset_mmdet_fold_0/annotations/validation_annotations.json',
    metric=[
        'bbox',
        'segm',
    ],
    dilate_mask=False)
custom_hooks = [
    dict(
        type='ModelCheckpointingHook',
        interval=1,
        metrics_file_name='metrics_fold_0_run_upernet_swin_dataset1_and_2.txt',
        chkp_dir=
        '/home/ec2-user/hubmap-hacking-the-human-vasculature/project_mmdet/models_fold_0_run_upernet_swin_dataset1_and_2',
        chkp_name='model_fold_0_run_upernet_swin_dataset1_and_2.pth',
        tgt_metric='coco/segm_mAP',
        should_record_epoch=False),
]
launcher = 'none'
work_dir = './work_dirs/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512'
