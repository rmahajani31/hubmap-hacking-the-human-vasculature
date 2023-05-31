export SUFFIX=all_valid_preproc
export BOUNDING_BOXES=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/segmentation_bbox_${SUFFIX}.csv
export IMAGE_LABELS=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/segmentation_labels_${SUFFIX}.csv
export INPUT_PREDICTIONS=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/seg_preds_${SUFFIX}.csv
export INSTANCE_SEGMENTATIONS=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/segmentation_masks_${SUFFIX}.csv
export OUTPUT_METRICS=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/metrics_${SUFFIX}.txt

python3 /home/rmahajani31/Projects/models/research/object_detection/metrics/oid_challenge_evaluation.py \
    --input_annotations_boxes=${BOUNDING_BOXES} \
    --input_annotations_labels=${IMAGE_LABELS} \
    --input_class_labelmap=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/label_map.pbtxt \
    --input_predictions=${INPUT_PREDICTIONS} \
    --input_annotations_segm=${INSTANCE_SEGMENTATIONS} \
    --output_metrics=${OUTPUT_METRICS}