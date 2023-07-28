_base_ = [
    './co_dino_5scale_r50_1x_coco.py'
]
 
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=36)


pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'

# model settings
model = dict(
    with_attn_mask=False,
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        drop_path_rate=0.5,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        out_indices=(0, 1, 2, 3),
        window_size=12,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained),
    neck=dict(in_channels=[192, 192*2, 192*4, 192*8]),
    query_head=dict(
        transformer=dict(
            encoder=dict(
                # number of layers that use checkpoint
                with_cp=6))))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (768, 768)
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

load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=image_size,
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
]

train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
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
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + train_ann_file,
            img_prefix=data_root + train_data_prefix,
            pipeline=load_pipeline),
        pipeline=train_pipeline),
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

load_from='/home/ec2-user/pretrained/co_dino_5scale_lsj_swin_large_3x_coco.pth'