# Imports
import sys
import logging
import os
from collections import OrderedDict
import torch
import shutil
from torch.nn.parallel import DistributedDataParallel
from sklearn.model_selection import KFold
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
import cv2
import pickle
from detectron2.structures import BoxMode
from detectron2.data import DatasetMapper
import detectron2.data.transforms as T
from detectron2 import model_zoo
import numpy as np


# In[2]:


# Custom HubMap Dataset
class HubMapDataset:
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_file_list = sorted(os.listdir(self.image_dir))
        self.annotation_file_list = os.listdir(self.annotation_dir)
        
    def __len__(self):
        return len(self.image_file_list)
        
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_file_list[idx])
        img_id = self.image_file_list[idx].split('.png')[0]
        annotation_path = os.path.join(self.annotation_dir, f'{img_id}.pkl')
        
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        record = {
            'file_name': image_path,
            'image_id': idx,
            'height': height,
            'width': width,
        }
        
        with open(annotation_path, 'rb') as f:
            orig_annots = pickle.load(f)
        
        objs = []
        for orig_annot in orig_annots:
            bbox = orig_annot['bbox']
            orig_annot['bbox'] = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            orig_annot['bbox_mode'] = BoxMode.XYXY_ABS
            objs.append(orig_annot)
            
        record['annotations'] = objs
        
        return record


# Function to register a dataset
CLASSES = ['blood_vessel']
def register_custom_dataset(dataset_name, image_dir, annotation_dir):
    DatasetCatalog.register(dataset_name, lambda: HubMapDataset(image_dir, annotation_dir))
    MetadataCatalog.get(dataset_name).set(thing_classes=CLASSES, evaluator_type="coco")


# In[5]:


class CustomArguments:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# In[6]:


# arguments
num_folds = 5
config_file = '/home/ec2-user/hubmap-hacking-the-human-vasculature/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
base_dataset_path = '/home/ec2-user/hubmap-hacking-the-human-vasculature/dataset1_files'
base_dataset_name = 'hubmap-dataset1'
num_machines = 1
num_gpus = 1
machine_rank = 0
port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
dist_url = "tcp://127.0.0.1:{}".format(port)
opts = []
# argument_dict = {'config_file':config_file, 'train_dataset_name': train_dataset_name, 'train_dir': train_dir, 'num_machines':num_machines, 'num_gpus':num_gpus, 'machine_rank': machine_rank, 'dist_url':dist_url, 'opts': opts}
# args = CustomArguments(**argument_dict)


# In[7]:


# Setup the config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.DATASETS.TRAIN = ()
cfg.DATASETS.TEST = ()
# cfg.INPUT.MIN_SIZE_TRAIN = (256,350,480,512)  # Minimum input image size during training
cfg.INPUT.MIN_SIZE_TRAIN = (350, 480, 512, 672, 736, 800)
cfg.INPUT.MAX_SIZE_TRAIN = 800     # Maximum input image size during training
cfg.INPUT.MIN_SIZE_TEST = (512,)      # Minimum input image size during testing
cfg.INPUT.MAX_SIZE_TEST = 512      # Maximum input image size during testing
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
# cfg.MODEL.WEIGHTS = '/home/ec2-user/hubmap-hacking-the-human-vasculature/project_detectron2/output/inference/best_model_fold_0_with_added_aug_lr_0.00025.pth'
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.00025
# cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = 6000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


# In[8]:


# compared to "train_net.py", we do not support accurate timing and
# precise BN here, because they are not trivial to implement in a small training loop
prob = 0.5
data_transforms = [
    T.RandomApply(T.RandomRotation([-90,90], expand=False), prob=prob),
    T.RandomFlip(horizontal=True, vertical=False, prob=prob),
    T.RandomFlip(horizontal=False, vertical=True, prob=prob),
    T.RandomApply(T.RandomBrightness(0.8, 1.2), prob=prob),
    T.RandomApply(T.RandomContrast(0.8, 1.2), prob=prob),
    T.RandomApply(T.RandomSaturation(0.8,1.2), prob=prob),
    T.RandomApply(T.RandomCrop('relative', (0.8, 0.8)), prob=prob)
]

# data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=data_transforms))


# In[9]:


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)



import time

max_iter = cfg.SOLVER.MAX_ITER
writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

output_dir = os.path.join(cfg.OUTPUT_DIR, "inference")

for i in range(num_folds):
    if os.path.exists(f'{output_dir}/model_stats_detectron_dataset1_fold_{i}.txt'):
        os.remove(f'{output_dir}/model_stats_detectron_dataset1_fold_{i}.txt')

for i in range(num_folds):
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "inference", f'{base_dataset_name}-train-fold-{i}')):
        shutil.rmtree(os.path.join(cfg.OUTPUT_DIR, "inference", f'{base_dataset_name}-train-fold-{i}'))
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, "inference", f'{base_dataset_name}-validation-fold-{i}')):
        shutil.rmtree(os.path.join(cfg.OUTPUT_DIR, "inference", f'{base_dataset_name}-validation-fold-{i}'))

for i in range(num_folds):
    register_custom_dataset(f'{base_dataset_name}-train-fold-{i}', f'{base_dataset_path}/all_dataset1_imgs_merged_train_{i}', f'{base_dataset_path}/all_dataset1_annotations_merged_train_{i}')
    register_custom_dataset(f'{base_dataset_name}-validation-fold-{i}', f'{base_dataset_path}/all_dataset1_imgs_merged_validation_{i}', f'{base_dataset_path}/all_dataset1_annotations_merged_validation_{i}')

for i in range(num_folds):
    train_dataset = DatasetCatalog.get(f'{base_dataset_name}-train-fold-{i}')
    train_data_loader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=data_transforms), dataset=train_dataset)
    validation_data_loader = build_detection_test_loader(cfg, f'{base_dataset_name}-validation-fold-{i}')
    evaluator = get_evaluator(
        cfg, f'{base_dataset_name}-validation-fold-{i}', os.path.join(cfg.OUTPUT_DIR, "inference", f'{base_dataset_name}-validation-fold-{i}')
    )
    
    model = build_model(cfg)
    model.train()
    resume = False
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    iterations_per_epoch = len(train_dataset) // cfg.SOLVER.IMS_PER_BATCH
    num_epochs = max_iter // iterations_per_epoch
    num_iterations_to_show_stats = 50
    max_ap_50 = 0
    loss_stats = {'total_loss': [], 'loss_cls': [], 'loss_box_reg': [], 'loss_mask': [], 'loss_rpn_cls': [], 'loss_rpn_loc': []}
    with open(f'{output_dir}/model_stats_detectron_dataset1_fold_{i}.txt', 'a') as f:
        f.write(f'Epoch info is - num_epochs: {num_epochs}, max_iter: {max_iter}, train_dataset_len: {len(train_dataset)}, iterations_per_epoch: {iterations_per_epoch}, num_iterations_to_show_stats: {num_iterations_to_show_stats}\n')
    # Training Loop
    with EventStorage(start_iter) as storage:
        start_time = time.time()
        for data, iteration in zip(train_data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            for loss_key in loss_dict:
                if loss_key in loss_stats:
                    loss_stats[loss_key].append(loss_dict[loss_key].item())
            losses = sum(loss_dict.values())
            loss_stats['total_loss'].append(losses.item())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
        
            if (iteration+1) % num_iterations_to_show_stats == 0:
                metrics = inference_on_dataset(model, validation_data_loader, evaluator)
                print('===========')
                print(metrics)
                print('===========')
                metrics_str = ''
                for task, task_metrics in metrics.items():
                    task_str = f'{task}: '
                    for metric, value in task_metrics.items():
                        task_str += f'{metric}={value:.4f}, '
                    metrics_str += task_str.rstrip(', ') + '\n'
                metrics_str = f'Iteration: {iteration}, time_taken: {float(time.time()-start_time)/60} minutes --> {metrics_str}'
                loss_str = ''
                for loss_key in loss_stats.keys():
                    loss_str += f'{loss_key} - {np.mean(loss_stats[loss_key])}, '
                    loss_stats[loss_key] = []
                if 'segm' in metrics and metrics['segm']['AP50'] > max_ap_50:
                    max_ap_50 = metrics['segm']['AP50']
                    torch.save(model.state_dict(), f'{output_dir}/best_model_fold_{i}.pth')
                    with open(f'{output_dir}/best_model_stats_detectron_dataset1_fold_{i}.txt', 'w') as f:
                        f.write(f'{metrics_str}\n{loss_str}\n')
                with open(f'{output_dir}/model_stats_detectron_dataset1_fold_{i}.txt', 'a') as f:
                    f.write(f'{metrics_str}\n{loss_str}\n')
                start_time = time.time()