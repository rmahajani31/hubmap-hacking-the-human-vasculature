if True:
    from hubmap_modules import *
    input_preds_files = ['val_preds_cascade_mask_rcnn.pkl', 'val_preds_effdet.pkl', 'val_preds_rtm.pkl', 'val_preds_yolo.pkl']
    input_pred_weights = [1, 1, 1, 1]
    ann_file = '/home/ec2-user/hubmap-hacking-the-human-vasculature/all_dataset_files/all_dataset_mmdet_fold_0/annotations/validation_annotations.json'
    proposal_nums=(1000, 1, 10)
    metric='bbox'
    met = HubMapDetEnsembleCocoMetric(ann_file=ann_file, proposal_nums=proposal_nums, metric=metric, input_preds_files=input_preds_files, input_pred_weights=input_pred_weights)
    met.compute_metrics([])