from mmseg.registry import DATASETS, TRANSFORMS
from mmengine.dataset import BaseDataset
from mmdet.structures.bbox import bbox_overlaps

from pycocotools.coco import COCO
import os
import pickle

def assign_gt(pr_bboxes, gt_bboxes):
    ious = bbox_overlaps(
        torch.from_numpy(pr_bboxes), torch.from_numpy(gt_bboxes)
    ).numpy()
    return ious.argmax(1)

@DATASETS.register_module()
class HubMapSegTrainDataset(BaseDataset):

    METAINFO = dict(
        classes=('unlabelled', 'blood_vessel',),
        palette=[[128, 64, 128], [244, 35, 232]])

    def __init__(self, data_root='', ann_file='', img_dir='', pipeline=None):
        self.img_dir = img_dir
        super(HubMapSegTrainDataset, self).__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline)
    
    def load_data_list(self):
        coco = COCO(os.path.join(self.data_root, self.ann_file))
        img_ids = coco.getImgIds()
        box_infos = []
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            anns = coco.loadAnns(coco.getAnnIds(img_id))
            for ann in anns:
                x1, y1, w, h = ann['bbox']
                box_infos.append(
                    dict(
                        filename=img_info['file_name'],
                        img_path=os.path.join(self.data_root, self.img_dir, img_info['file_name']),
                        bbox=[x1, y1, x1 + w, y1 + h],
                        segmentation=ann['segmentation'][0],
                        score=1.0,  # GT
                        category_id=ann['category_id'],
                        img_id=img_id,
                        height=img_info['height'],
                        width=img_info['width'],
                    )
                )
        return box_infos

@DATASETS.register_module()
class HubMapSegTestDataset(BaseDataset):
    METAINFO = dict(
        classes=('unlabelled', 'blood_vessel',),
        palette=[[128, 64, 128], [244, 35, 232]])

    def __init__(self, preds_file='', pipeline=None):
        # super().__init__(pipeline=pipeline)
        self.preds_file = preds_file
        super(HubMapSegTestDataset, self).__init__(pipeline=pipeline)
        # box_infos = []
        # coco = COCO(ann_file)
        # with open(self.preds_file, 'rb') as f:
        #     bbox_preds = pickle.load(f)
        # for bbox_pred in bbox_preds:
        #     img_id = bbox_pred['img_id']
        #     img_path = bbox_pred['img_path']
        #     height, width = bbox_pred['ori_shape']
        #     pred_instances = bbox_pred['pred_instances']
        #     labels = pred_instances['labels']
        #     pr_bboxes = pred_instances['bboxes']
        #     scores = pred_instances['scores']
            
        #     img_info = coco.loadImgs(img_id)[0]
        #     anns = coco.loadAnns(coco.getAnnIds(img_id))
        #     gt_bboxes = np.array([ann['bbox'] for ann in anns])
        #     gt_bboxes[:, 2:] += gt_bboxes[:, :2]  # xywh2xyxy

        #     gt_inds = assign_gt(pr_bboxes, gt_bboxes)

    def load_data_list(self):
        box_infos = []
        with open(self.preds_file, 'rb') as f:
            bbox_preds = pickle.load(f)
        for bbox_pred in bbox_preds:
            img_id = bbox_pred['img_id']
            img_path = bbox_pred['img_path']
            # height, width = bbox_pred['ori_shape']
            height, width = (512, 512)
            # pred_instances = bbox_pred['pred_instances']
            # labels = pred_instances['labels']
            # bboxes = pred_instances['bboxes']
            # scores = pred_instances['scores']
            labels = bbox_pred['labels']
            bboxes = bbox_pred['bboxes']
            scores = bbox_pred['scores']
            for i in range(len(bboxes)):
                label = labels[i].item()
                bbox = bboxes[i].tolist()
                score = scores[i].item()
                x1,y1,x2,y2 = bbox
                box_infos.append(
                    dict(
                        img_path=img_path,
                        bbox=[x1, y1, x2, y2],
                        score=score,
                        category_id=label,
                        img_id=img_id,
                        height=height,
                        width=width,
                    )
                )
        return box_infos

