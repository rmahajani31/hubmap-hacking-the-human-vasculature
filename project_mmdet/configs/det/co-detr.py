_base_ = [
    './co_dino_5scale_r50_1x_coco.py'
]
 
optimizer = dict(weight_decay=0.05)
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=36)


pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        out_indices=(1, 2, 3),
        window_size=7,
        ape=False,
        drop_path_rate=0.4,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained),
    neck=dict(in_channels=[96*2, 96*4, 96*8]))

# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='SwinTransformer',
#         embed_dims=96,
#         depths=[2, 2, 18, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         drop_path_rate=0.4,
#         patch_norm=True,
#         out_indices=(1, 2, 3),
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     neck=dict(in_channels=[96*2, 96*4, 96*8])
# )

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024, 1024)
# load_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(
#         type='Resize',
#         img_scale=image_size,
#         ratio_range=(0.1, 2.0),
#         multiscale_mode='range',
#         keep_ratio=True),
#     dict(
#         type='RandomCrop',
#         crop_type='absolute_range',
#         crop_size=image_size,
#         recompute_bbox=True,
#         allow_negative_crop=True),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
# ]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=image_size, keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


dataset_type = 'CocoDataset'
generate_all_datset_annots = True
generate_pseudo_labels = False
pseudo_thresh=0.3
base_data_dir_name_1 = 'dataset1_files' if not generate_all_datset_annots else 'all_dataset_files'
base_data_dir_name_2 = 'all_dataset1' if not generate_all_datset_annots else 'all_dataset'
pseudo_label_name = '' if not generate_pseudo_labels else '_pseudo_labels'
suffix_end = 'only_dataset1' if not generate_all_datset_annots else 'dataset1_and_2'
suffix_end = suffix_end if not generate_pseudo_labels else f'pseudo_label_{pseudo_thresh}'
suffix = f'fold_0_run_codetr_{suffix_end}'

chkp_dir = f'/home/ec2-user/hubmap-hacking-the-human-vasculature/project_mmdet/models_{suffix}'
metrics_file_name = f'metrics_{suffix}.txt'
chkp_name = f'model_{suffix}.pth'

# Path of train annotation file
train_ann_file = 'annotations/train_annotations.json'
train_data_prefix = 'train_images/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/validation_annotations.json'
val_data_prefix = 'validation_images/'  # Prefix of val image path

classes = ('blood_vessel',)
num_classes = len(classes)  # Number of classes for classification


data_root = f'/home/ec2-user/hubmap-hacking-the-human-vasculature/{base_data_dir_name_1}/{base_data_dir_name_2}{pseudo_label_name}_mmdet_fold_1/'
dataset_type = 'CocoDataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + train_ann_file,
            img_prefix=data_root + train_data_prefix,
            pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + val_ann_file,
        img_prefix=data_root + val_data_prefix,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + val_ann_file,
        img_prefix=data_root + val_data_prefix,
        pipeline=test_pipeline))

load_from='/home/ec2-user/pretrained/co_deformable_detr_swin_small_3x_coco.pth'