export BOUNDING_BOXES=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/segmentation_bbox.csv
export IMAGE_LABELS=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/segmentation_labels.csv
export INPUT_PREDICTIONS=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/seg_preds.csv
export INSTANCE_SEGMENTATIONS=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/segmentation_masks.csv
export OUTPUT_METRICS=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/metrics.txt

python3 object_detection/metrics/oid_challenge_evaluation.py \
    --input_annotations_boxes=${BOUNDING_BOXES} \
    --input_annotations_labels=${IMAGE_LABELS} \
    --input_class_labelmap=/home/rmahajani31/Projects/hubmap-hacking-the-human-vasculature/map_input_data/label_map.pbtxt \
    --input_predictions=${INPUT_PREDICTIONS} \
    --input_annotations_segm=${INSTANCE_SEGMENTATIONS} \
    --output_metrics=${OUTPUT_METRICS}
