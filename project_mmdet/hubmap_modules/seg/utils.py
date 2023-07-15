import numpy as np
import cv2

def to_mask(mask_ann, img_h, img_w):
    mask = np.zeros((img_h, img_w))
    coords = [[x, y] for x, y in zip(mask_ann[::2], mask_ann[1::2])]
    cv2.fillPoly(mask, pts=[np.array(coords)], color=1)
    # mask = mask - 1
    # mask[mask == -1] = 255
    return mask.astype(np.uint8)