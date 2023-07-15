from mmseg.registry import DATASETS, TRANSFORMS
from mmengine.dataset import BaseDataset

from pycocotools.coco import COCO
import os

@DATASETS.register_module()
class HubMapSegDataset(BaseDataset):

    METAINFO = dict(
        classes=('blood_vessel',),
        palette=[[128, 64, 128]])

    def __init__(self, data_root='', ann_file='', img_dir='', pipeline=None):
        self.img_dir = img_dir
        super(HubMapSegDataset, self).__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline)
    
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

