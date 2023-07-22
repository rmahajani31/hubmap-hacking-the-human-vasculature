from hubmap_modules import *
import torch
import numpy as np
import cv2
import pickle

def generate_ensemble_preds(input_preds_files, input_pred_weights, ann_file='/home/ec2-user/hubmap-hacking-the-human-vasculature/all_dataset_files/all_dataset_mmdet_fold_0/annotations/validation_annotations.json', proposal_nums=(1000, 1, 10), metric='bbox'):
    met = HubMapDetEnsembleCocoMetric(ann_file=ann_file, proposal_nums=proposal_nums, metric=metric, input_preds_files=input_preds_files, input_pred_weights=input_pred_weights)
    met.compute_metrics([])