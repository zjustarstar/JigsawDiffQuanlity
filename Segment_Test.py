import numpy as np
import os
import tqdm
from ultralytics import SAM
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt

from segment_anything.utils.amg import  area_from_rle


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask

    plt.subplot(1,2,2)
    plt.imshow(img)


def seg_by_fb_sam(img_path):
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
    sam.to('cuda')
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    # cv2.imshow('original', resized)



    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=36,# 32 86  10 24  随机点,该值越大，分割结果越细
        pred_iou_thresh=0.92,
        # crop_overlap_ratio=0.1,
        min_mask_region_area=10000 #这个参数不能用来削减mask的数量，只能削减分离的mask，如一个mask中被截断，小部分的就舍去
    )

    masks = mask_generator.generate(image)

    # for mask_data in masks:
    #
    #
    #     # 计算 RLE 表示的遮罩的面积
    #
    #     print("Mask area:",mask_data['area'])

    plt.figure(figsize=(20, 20))
    plt.subplot(1,2,1)
    plt.imshow(image)
    show_anns(masks)

    # save fip
    path, fname = os.path.split(img_path)
    name, exe = os.path.splitext(fname)
    newname = os.path.join(path, name+"_mask_24_092_default" + exe)
    #plt.savefig(newname)
    #plt.close()

    plt.axis('off')
    plt.show()

    return image


if __name__ == '__main__':
    # Hyperparameters
    data_path = 'testimg\\2'
    puzzle_size = (10, 10) # (num_puzzle_h:int,num_puzzle_w:int)

    #model = SAM('models/sam_b.pt')
    img_list = [i for i in os.listdir(path=data_path) if i.endswith('.jpg')]
    non_border_patch_size = (puzzle_size[0]-2, puzzle_size[1]-2)
    for idx,img_name in tqdm.tqdm(enumerate(img_list)):

        img_path = os.path.join(data_path,img_name)

        seg_by_fb_sam(img_path)

