_base_ = [
    '/home/ec2-user/mmsegmentation/configs/_base_/default_runtime.py', '/home/ec2-user/mmsegmentation/configs/_base_/schedules/schedule_40k.py'
]

configs=['/home/ec2-user/hubmap-hacking-the-human-vasculature/project_mmdet/configs/seg/segformer.py', '/home/ec2-user/hubmap-hacking-the-human-vasculature/project_mmdet/configs/seg/upernet_vit.py']
checkpoints=['/home/ec2-user/hubmap-hacking-the-human-vasculature/project_mmdet/seg_models/model_fold_0_run_segformer_b4_dataset1_and_2_0.601.pth','/home/ec2-user/hubmap-hacking-the-human-vasculature/project_mmdet/seg_models/model_fold_0_run_upernet_vit_dataset1_and_2_0.63.pth']
weights=[0.5,0.5]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
classes = ('unlabelled', 'blood_vessel')
train_cfg = dict(val_interval=500)
input_size = (224, 224)
data_preprocessor = dict(
    type='CustomSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=input_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(type='EnsembleSegmentor',
             data_preprocessor=data_preprocessor,
             configs=configs,
             weights=weights,
             checkpoints=checkpoints,
             train_cfg=train_cfg,
             test_cfg=dict(mode='whole'))

dataset_type = 'HubMapSegTrainDataset'
generate_all_datset_annots = True
base_data_dir_name_1 = 'dataset1_files' if not generate_all_datset_annots else 'all_dataset_files'
base_data_dir_name_2 = 'all_dataset1' if not generate_all_datset_annots else 'all_dataset'
data_root = f'/home/ec2-user/hubmap-hacking-the-human-vasculature/{base_data_dir_name_1}/{base_data_dir_name_2}_mmdet_fold_0/'
suffix_end = 'only_dataset1' if not generate_all_datset_annots else 'dataset1_and_2'
suffix = f'fold_0_run_segformer_b4_{suffix_end}'

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

# val_dataloader = dict(
#     batch_size=64,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=val_ann_file,
#         img_dir=val_data_prefix,
#         pipeline=test_pipeline))

val_dataloader = dict(
    batch_size=1,
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
