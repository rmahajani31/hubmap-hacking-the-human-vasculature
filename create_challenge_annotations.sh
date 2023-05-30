export HIERARCHY_FILE=/home/rmahajani31/Projects/models/research/object_detection/challenge_data/challenge-2019-label300-segmentable-hierarchy.json
export BOUNDING_BOXES=/home/rmahajani31/Projects/models/research/object_detection/challenge_data/challenge-2019-validation-segmentation-bbox
export IMAGE_LABELS=/home/rmahajani31/Projects/models/research/object_detection/challenge_data/challenge-2019-validation-segmentation-labels
export INSTANCE_SEGMENTATIONS=/home/rmahajani31/Projects/models/research/object_detection/challenge_data/challenge-2019-validation-segmentation-masks-formatted

python /home/rmahajani31/Projects/models/research/object_detection/dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${BOUNDING_BOXES}.csv \
    --output_annotations=${BOUNDING_BOXES}_expanded.csv \
    --annotation_type=1

python /home/rmahajani31/Projects/models/research/object_detection/dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${IMAGE_LABELS}.csv \
    --output_annotations=${IMAGE_LABELS}_expanded.csv \
    --annotation_type=2

python /home/rmahajani31/Projects/models/research/object_detection/dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${INSTANCE_SEGMENTATIONS}.csv \
    --output_annotations=${INSTANCE_SEGMENTATIONS}_expanded.csv \
    --annotation_type=1
