#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch
import segmentation_models_pytorch as smp


# In[2]:


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[3]:


DATA_DIR = './'
x_train_dir = os.path.join(DATA_DIR, 'dataset1_files/all_dataset1_imgs_merged_train_0')
y_train_dir = os.path.join(DATA_DIR, 'dataset1_files/all_dataset1_masks_merged_train_0')

x_valid_dir = os.path.join(DATA_DIR, 'dataset1_files/all_dataset1_imgs_merged_validation_0')
y_valid_dir = os.path.join(DATA_DIR, 'dataset1_files/all_dataset1_masks_merged_validation_0')

x_test_dir = os.path.join(DATA_DIR, 'dataset1_files/all_dataset1_imgs_merged_validation_0')
y_test_dir = os.path.join(DATA_DIR, 'dataset1_files/all_dataset1_masks_merged_validation_0')


# In[4]:


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# In[5]:


class HubMapDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['blood_vessel']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return self.images_fps[i], image, mask.astype('int')
        
    def __len__(self):
        return len(self.ids)


# In[6]:


def get_training_augmentation_simple():
  train_transform = [
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
  ]
  return albu.Compose(train_transform)
  
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=512, min_width=352, always_apply=True, border_mode=0),
        albu.RandomCrop(height=512, width=352, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)


# In[7]:


# In[8]:


CLASSES = ['blood_vessel']
ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'


# In[9]:


model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=len(CLASSES)
)
model = model.to(DEVICE)


# In[10]:


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


# In[11]:


train_dataset = HubMapDataset(
    x_train_dir, 
    y_train_dir,
    augmentation=get_training_augmentation_simple(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = HubMapDataset(
    x_valid_dir, 
    y_valid_dir, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)


# In[12]:


from torchmetrics import Metric
class IoUScore(Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("iou_scores_back", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("iou_scores_fore", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = (preds > self.threshold).int()
        intersection_back = torch.logical_and(preds==0, target==0).sum()
        union_back = torch.logical_or(preds==0, target==0).sum()
        intersection_fore = torch.logical_and(preds==1, target==1).sum()
        union_fore = torch.logical_or(preds==1, target==1).sum()
        self.iou_scores_back = torch.cat((self.iou_scores_back, torch.tensor([intersection_back.float() / union_back.float()], device=DEVICE)))
        self.iou_scores_fore = torch.cat((self.iou_scores_fore, torch.tensor([intersection_fore.float() / union_fore.float()], device=DEVICE)))
    
    def interm_compute(self):
        iou_back = self.iou_scores_back.mean().item()
        iou_fore = self.iou_scores_fore.mean().item()
        return iou_back, iou_fore
    
    def compute(self):
        iou_back = self.iou_scores_back.mean().item()
        iou_fore = self.iou_scores_fore.mean().item()
        self.iou_scores_back = torch.tensor([], device=DEVICE)
        self.iou_scores_fore = torch.tensor([], device=DEVICE)
        return iou_back,iou_fore


# In[13]:


loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
train_metric = IoUScore(threshold=0.5).to(DEVICE)
validation_metric = IoUScore(threshold=0.5).to(DEVICE)

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])


# In[17]:


from tqdm import tqdm
# Training loop
def train_epoch(model, train_metric, validation_metric, optimizer, device, dataloader, validation_loader, epoch, fp, max_iou):
    model.train()
    num_batches = len(dataloader)
    total_loss = 0
    gradient_accumulation_steps = 8
    print(f'Processing a total of {num_batches} batches in training')
    # Iterate over the training dataset
    for batch_idx, (f, inputs, targets) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        outputs = outputs.squeeze(dim=1)
        # print(f'Targets Shape: {targets.shape}, Outputs shape: {outputs.shape}')
        # print(targets.sum())
#         print(f'input and output shapes: {inputs.shape}, {outputs.shape}, {targets.shape}')
        # print(f'Outputs min: {torch.min(outputs)}, Outputs max: {torch.max(outputs)}')
        loss = loss_fn(outputs, targets)

        # Backward pass
        loss.backward()

        if (batch_idx+1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Compute metrics
        train_metric.update(torch.sigmoid(outputs), targets)
        
        total_loss += loss
        if (batch_idx+1) % 256 == 0 or (batch_idx+1) == num_batches:
            print('Computing Stats....')
            train_metric_vals = [float(total_loss)/(batch_idx+1)] + [train_metric.compute() if (batch_idx+1) == num_batches else train_metric.interm_compute()]
            validation_metric_vals = valid_epoch(model, validation_metric, device, validation_loader)
            model.train()
            cur_validation_iou = 0.5*validation_metric_vals[1][0] + 0.5*validation_metric_vals[1][1]
            if cur_validation_iou > max_iou and (cur_validation_iou-max_iou) >= 0.01:
                print(f'Saving model with IoU: {cur_validation_iou}...')
                torch.save(model, './models/best_model_unet_dataset1.pth')
                with open('./models/best_model_unet_dataset1.txt', 'w') as f:
                    f.write(f"Epoch {epoch} Iteration {batch_idx}: Train Loss={train_metric_vals[0]}, Validation Loss={validation_metric_vals[0]}, Train IoU Back={train_metric_vals[1][0]}, Train IoU Fore={train_metric_vals[1][1]}, Validation IoU Back={validation_metric_vals[1][0]}, Validation IoU Fore={validation_metric_vals[1][1]}")
                    max_iou = cur_validation_iou
            print(f"Epoch {epoch} Iteration {batch_idx}: Train Loss={train_metric_vals[0]}, Validation Loss={validation_metric_vals[0]}, Train IoU Back={train_metric_vals[1][0]}, Train IoU Fore={train_metric_vals[1][1]}, Validation IoU Back={validation_metric_vals[1][0]}, Validation IoU Fore={validation_metric_vals[1][1]}")
            fp.write(f"Epoch {epoch} Iteration {batch_idx}: Train Loss={train_metric_vals[0]}, Validation Loss={validation_metric_vals[0]}, Train IoU Back={train_metric_vals[1][0]}, Train IoU Fore={train_metric_vals[1][1]}, Validation IoU Back={validation_metric_vals[1][0]}, Validation IoU Fore={validation_metric_vals[1][1]}\n")
            fp.flush()
    return max_iou

def valid_epoch(model, validation_metric, device, dataloader):
    model.eval()
    num_batches = len(dataloader)
    total_loss = 0
    print(f'Processing a total of {num_batches} batches in validation')
    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over the validation dataset
        for batch_idx, (f, inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze(dim=1)
            loss = loss_fn(outputs, targets)
            # Compute metrics
            validation_metric.update(torch.sigmoid(outputs), targets)
            total_loss += loss
    # Get the metric values
    metric_values = [float(total_loss)/num_batches] + [validation_metric.compute()]
    return metric_values

# This is a block to run training without cross validation
import time
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2)
max_iou = 0
num_epochs = 50
if os.path.exists('./models/model_stats_unet_dataset1.txt'):
 os.remove('./models/model_stats_unet_dataset1.txt')
fp = open('./models/model_stats_unet_dataset1.txt', 'a')
for epoch in range(num_epochs):
   # Training
   start_time = time.time()
   max_iou = train_epoch(model, train_metric, validation_metric, optimizer, DEVICE, train_loader, valid_loader, epoch, fp, max_iou)
   print(f'=========Finished Training Epoch {epoch} in {float(time.time()-start_time)/60}==========')
fp.close()