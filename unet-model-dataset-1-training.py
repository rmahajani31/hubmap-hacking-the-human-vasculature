#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch.nn as nn
import albumentations as albu
import torch
import segmentation_models_pytorch as smp


# In[4]:


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE


# In[22]:


DATA_DIR = './'
x_train_dir = os.path.join(DATA_DIR, 'dataset1_files/all_dataset1_imgs')
y_train_dir = os.path.join(DATA_DIR, 'dataset1_files/all_dataset1_masks')

x_valid_dir = os.path.join(DATA_DIR, 'dataset1_files/all_dataset2_imgs')
y_valid_dir = os.path.join(DATA_DIR, 'dataset1_files/all_dataset2_masks')


# In[23]:


len(os.listdir(x_train_dir)), len(os.listdir(y_train_dir)), len(os.listdir(x_valid_dir)), len(os.listdir(y_valid_dir))


# In[24]:


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


# In[25]:


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
    
    CLASSES = ['unlabelled', 'blood_vessel']
    
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
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return self.masks_fps[i], image, mask
        
    def __len__(self):
        return len(self.ids)


# In[26]:


def get_training_augmentation():
  train_transform = [
    albu.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=90),
    albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0),
    albu.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0),
    albu.Flip(),
    albu.RandomBrightnessContrast(),
    albu.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), p=1),
    albu.ColorJitter()
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
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# In[27]:


CLASSES = ['unlabelled', 'blood_vessel']
ENCODER = 'efficientnet-b7'
ENCODER_WEIGHTS = 'imagenet'


# In[28]:


# In[29]:


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


# In[30]:


train_dataset = HubMapDataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = HubMapDataset(
    x_valid_dir, 
    y_valid_dir, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)


# In[31]:


_, image, mask = train_dataset[0]
print(image.shape, mask.shape)


# In[32]:


from torchmetrics import Metric
class IoUScore(Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = threshold
        self.add_state("intersection_back", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("union_back", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("intersection_fore", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("union_fore", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_images", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = (preds > self.threshold).int()
        intersection_back = torch.logical_and(preds[:,0,:,:], target[:,0,:,:]).sum()
        union_back = torch.logical_or(preds[:,0,:,:], target[:,0,:,:]).sum()
        intersection_fore = torch.logical_and(preds[:,1,:,:], target[:,1,:,:]).sum()
        union_fore = torch.logical_or(preds[:,1,:,:], target[:,1,:,:]).sum()

        self.intersection_back += intersection_back
        self.union_back += union_back
        self.intersection_fore += intersection_fore
        self.union_fore += union_fore

    def compute(self):
        iou_back = (self.intersection_back.float() / self.union_back.float())
        iou_fore = (self.intersection_fore.float() / self.union_fore.float())
        self.intersection_back = 0
        self.union_back = 0
        self.intersection_fore = 0
        self.union_fore = 0
        return iou_back,iou_fore


# In[33]:


def dice_loss(preds, targets, class_weights, threshold=0.5, smooth=1e-5):
    preds_probs = torch.softmax(preds, dim=1)
    preds_flat = preds_probs.view(preds_probs.shape[0], preds_probs.shape[1], -1)
    targets_flat = targets.view(targets.shape[0], targets.shape[1], -1)
    intersection_vals = preds_flat * targets_flat
    intersection_sum = intersection_vals.sum(dim=(-1,0))
    denom_sum = preds_flat.sum(dim=(-1,0)) + targets_flat.sum(dim=(-1,0))
    dice_coeffs = (2 * intersection_sum + smooth) / (denom_sum + smooth)
    dice_coeff = torch.sum(dice_coeffs * class_weights) / torch.sum(class_weights)
    return 1 - dice_coeff


# In[34]:


import torchmetrics
metrics = [
    IoUScore(threshold=0.5).to(DEVICE),
]

# In[35]:


from tqdm import tqdm
# Training loop
def train_epoch(model, metrics, optimizer, device, dataloader, class_weights=[0.05,1]):
    model.train()
    num_batches = len(dataloader)
    total_loss = 0
    class_weights = torch.tensor(class_weights, dtype=torch.float32, requires_grad=False)
    class_weights = class_weights.to(device)
    print(f'Processing a total of {num_batches} batches in training')
    # Iterate over the training dataset
    for batch_idx, (f, inputs, targets) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
#         print(f'input and output shapes: {inputs.shape}, {outputs.shape}, {targets.shape}')
#         print(f'Outputs min: {torch.min(outputs)}, Outputs max: {torch.max(outputs)}')
        loss = dice_loss(outputs, targets, class_weights)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute metrics
        for metric in metrics:
            metric.update(torch.softmax(outputs, dim=1), targets)
        total_loss += loss
    # Get the metric values
    metric_values = [float(total_loss)/num_batches] + [metric.compute() for metric in metrics]
    return metric_values

def valid_epoch(model, metrics, device, dataloader, class_weights=[0.05,1]):
    model.eval()
    num_batches = len(dataloader)
    total_loss = 0
    class_weights = torch.tensor(class_weights, dtype=torch.float32, requires_grad=False)
    class_weights = class_weights.to(device)
    print(f'Processing a total of {num_batches} batches in validation')
    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over the validation dataset
        for batch_idx, (f, inputs, targets) in tqdm(enumerate(dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = dice_loss(outputs, targets, class_weights)

            # Compute metrics
            for metric in metrics:
                metric.update(torch.softmax(outputs, dim=1), targets)
            total_loss += loss

    # Get the metric values
    metric_values = [float(total_loss)/num_batches] + [metric.compute() for metric in metrics]
    return metric_values


# In[36]:


## This is a block to run training with cross validation
import time
from sklearn.model_selection import KFold
from torch.utils.data import Subset
num_folds = 5

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
num_epochs = 50

for i in range(0, num_folds):
  if os.path.exists(f'./models/model_stats_unet_dataset1_fold_{i}.txt'):
    os.remove(f'./models/model_stats_unet_dataset1_fold_{i}.txt')

for fold, (train_indices, valid_indices) in enumerate(kfold.split(train_dataset)):
  cur_train_dataset = Subset(train_dataset, train_indices)
  cur_valid_dataset = Subset(train_dataset, valid_indices)
  train_loader = DataLoader(cur_train_dataset, batch_size=4, shuffle=True, num_workers=2)
  valid_loader = DataLoader(cur_valid_dataset, batch_size=4, shuffle=False, num_workers=2)
  max_iou = 0
  print(f'Starting fold {fold} with dataset sizes: {len(cur_train_dataset)}, {len(cur_valid_dataset)}')
  data_parallel = False
  model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=len(CLASSES)
    )
  model = model.to(DEVICE)
  model = nn.DataParallel(model) if data_parallel else model
  optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
  ])
  num_continuous_unsaved_epochs = 0
  for epoch in range(num_epochs):
      # Training
      start_time = time.time()
      train_metrics = train_epoch(model, metrics, optimizer, DEVICE, train_loader)
      print(f'=========Finished Training Epoch {epoch} in {float(time.time()-start_time)/60}==========')
      # Validation
      start_time = time.time()
      valid_metrics = valid_epoch(model, metrics, DEVICE, valid_loader)
      print(f'=========Finished Validation Epoch {epoch} {float(time.time()-start_time)/60}in =========')

      cur_validation_iou = 0.5*valid_metrics[1][0] + 0.5*valid_metrics[1][1]
      if cur_validation_iou > max_iou:
        print(f'Saving model with IoU: {cur_validation_iou}...')
        torch.save(model, f'./models/best_model_unet_dataset1_fold_{fold}.pth')
        with open(f'./models/best_model_unet_dataset1_fold_{fold}.txt', 'w') as f:
          f.write(f"Epoch {epoch}: Train Loss={train_metrics[0]}, Validation Loss={valid_metrics[0]}, Train IoU Back={train_metrics[1][0]}, Train IoU Fore={train_metrics[1][1]}, Validation IoU Back={valid_metrics[1][0]}, Validation IoU Fore={valid_metrics[1][1]}")
        max_iou = cur_validation_iou
        num_continuous_unsaved_epochs = 0
      else:
        num_continuous_unsaved_epochs += 1
      # Print or log the metrics for each epoch
      print(f"Epoch {epoch}: Train Loss={train_metrics[0]}, Validation Loss={valid_metrics[0]}, Train IoU Back={train_metrics[1][0]}, Train IoU Fore={train_metrics[1][1]}, Validation IoU Back={valid_metrics[1][0]}, Validation IoU Fore={valid_metrics[1][1]}")
      with open(f'./models/model_stats_unet_dataset1_fold_{fold}.txt', 'a') as fp:
        fp.write(f"Epoch {epoch}: Train Loss={train_metrics[0]}, Validation Loss={valid_metrics[0]}, Train IoU Back={train_metrics[1][0]}, Train IoU Fore={train_metrics[1][1]}, Validation IoU Back={valid_metrics[1][0]}, Validation IoU Fore={valid_metrics[1][1]}\n")
        fp.flush()
      if num_continuous_unsaved_epochs == 7:
          break



# In[20]:


## This is a block to run training without cross validation
#import time
#train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
#valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2)
#max_iou = 0
#num_epochs = 50
#if os.path.exists('./models/model_stats_unet_dataset1.txt'):
#  os.remove('./models/model_stats_unet_dataset1.txt')
#fp = open('./models/model_stats_unet_dataset1.txt', 'a')
#for epoch in range(num_epochs):
#    # Training
#    start_time = time.time()
#    train_metrics = train_epoch(model, metrics, optimizer, DEVICE, train_loader)
#    print(f'=========Finished Training Epoch {epoch} in {float(time.time()-start_time)/60}==========')
#    # Validation
#    start_time = time.time()
#    valid_metrics = valid_epoch(model, metrics, DEVICE, valid_loader)
#    print(f'=========Finished Validation Epoch {epoch} {float(time.time()-start_time)/60}in =========')
#    
#    save_interval = 10
#    if (epoch+1) % 10 == 0:
#        torch.save(model, f'./models/model_{epoch}_unet_dataset1.pth')
#    
#    cur_validation_iou = 0.5*valid_metrics[1][0] + 0.5*valid_metrics[1][1]
#    if cur_validation_iou > max_iou:
#      print(f'Saving model with IoU: {cur_validation_iou}...')
#      torch.save(model, './models/best_model_unet_dataset1.pth')
#      with open('./models/best_model_unet_dataset1.txt', 'w') as f:
#        f.write(f"Epoch {epoch}: Train Loss={train_metrics[0]}, Validation Loss={valid_metrics[0]}, Train IoU Back={train_metrics[1][0]}, Train IoU Fore={train_metrics[1][1]}, Validation IoU Back={valid_metrics[1][0]}, Validation IoU Fore={valid_metrics[1][1]}")
#      max_iou = cur_validation_iou
    # Print or log the metrics for each epoch
#    print(f"Epoch {epoch}: Train Loss={train_metrics[0]}, Validation Loss={valid_metrics[0]}, Train IoU Back={train_metrics[1][0]}, Train IoU Fore={train_metrics[1][1]}, Validation IoU Back={valid_metrics[1][0]}, Validation IoU Fore={valid_metrics[1][1]}")
#    fp.write(f"Epoch {epoch}: Train Loss={train_metrics[0]}, Validation Loss={valid_metrics[0]}, Train IoU Back={train_metrics[1][0]}, Train IoU Fore={train_metrics[1][1]}, Validation IoU Back={valid_metrics[1][0]}, Validation IoU Fore={valid_metrics[1][1]}\n")
#    fp.flush()
#fp.close()


# In[ ]:




