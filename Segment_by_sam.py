import numpy as np
import os
import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask

    # plt.subplot(1,2,2)
    # plt.imshow(img)
    return img


def load_sam_model():
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
    sam.to('cuda')
    return sam


def seg_by_fb_sam(sam, image, area_thre=20*20):
    """

    :param image: return of cv2.imread()
    :param sam: segmentation model
    :return: image:
             F_result:
             img:
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,# 32 86  10 24  随机点,该值越大，分割结果越细
        pred_iou_thresh=0.92,
        # crop_overlap_ratio=0.1,
        min_mask_region_area=10000 #这个参数不能用来削减mask的数量，只能削减分离的mask，如一个mask中被截断，小部分的就舍去
    )

    sam_result = mask_generator.generate(image)

    F_result = []
    for i in range(len(sam_result)):
        if sam_result[i]["area"] > area_thre:
            F_result.append(sam_result[i])

    seg_img = show_anns(F_result)
    return F_result, seg_img



if __name__ == '__main__':
    # Hyperparameters
    data_path = 'testimg\\1'
    puzzle_size = (10, 10) # (num_puzzle_h:int,num_puzzle_w:int)

    img_list = [i for i in os.listdir(path=data_path) if i.endswith('.jpg')]
    non_border_patch_size = (puzzle_size[0]-2, puzzle_size[1]-2)

    sam = load_sam_model()
    for idx,img_name in tqdm.tqdm(enumerate(img_list)):

        img_path = os.path.join(data_path,img_name)
        image = cv2.imread(img_path)

        seg_by_fb_sam(sam, image)

