from hubmap_modules import *

data_root = '/home/ec2-user/hubmap-hacking-the-human-vasculature/all_dataset_files/all_dataset_mmdet_fold_0/'
ann_file = 'annotations/train_annotations.json'
img_dir = 'train_images/'
train_pipeline = [
    dict(type='LoadBiomedicalImageFromFile'),
    dict(type='LoadSegMask'),
    dict(type='BoxJitter'),
    dict(type='ROIAlign', output_size=(128, 128)),
    dict(type='FormatSegMask'),
    dict(type='PackSegInputs')]

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadSegMask')]

dataset = HubMapSegDataset(data_root=data_root, ann_file=ann_file, img_dir=img_dir, pipeline=train_pipeline)

print(dataset[0])
