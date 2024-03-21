""" this is an object detection,use yolov8,
    path=./models/yolov8n.pt
    return a label list
    """

import numpy as np
from segment_anything import SamAutomaticMaskGenerator,sam_model_registry
import clip
import math


def check_include(boxes,box,block_label,n,block_w,block_h):
    """


    :param boxes:
    :param box:
    :param block_label:
    :param n:
    :param block_w:
    :param block_h:
    :return:
    """
    for i, box_list in enumerate(box):
        x1, y1, x2, y2 = (val for val in box[i])
        start = []
        end = []
        flag = True
        for row_s in range(n+1):
            if row_s*block_h > y1:
                start.append(row_s)
                for col_s in range(n+1):
                    if col_s*block_w > x1:
                        start.append(col_s)
                        flag=False
                        break
            if flag == False:
                break

        flag = True
        for row_e in range(start[0], n+1):
            if row_e*block_h >= y2 or row_e == n:
                end.append(row_e)
                for col_e in range(start[1],n+1):
                    if col_e*block_w > x2 or col_e==n:
                        end.append(col_e)
                        flag=False
                        break
            if flag==False:
                break

        for row in range(start[0],end[0]+1):
            for col in range(start[1],end[1]+1):
                block_label[row-1][col-1]=boxes.cls[i]+1
        # print(f"block_label:{block_label}")
    return block_label


def note_object_labels(yolo_model,img,n):




    h,w,d=img.shape
    block_h=h//n
    block_w=w//n

    results = yolo_model.predict(img)

    label_list=[]
    for result in results:

        box=result.boxes.xyxy
        blocks_label=np.zeros((n,n))
        blocks_label=check_include(result.boxes,box,blocks_label,n,block_w,block_h)
        label_list.append(blocks_label)

    return label_list,results


def note_sam_object_label(image,masks,n,maximum):
    """

    :param image: the picture
    :param masks: use sam_generator's masks
    :param n: get image's size
    :return: block_list: each part of image n*n  ;  sam_laebl : object 100%
    """
    w,h,_=image.shape
    sam_blocks=[]

    sam_label=np.zeros((n,n))
    blocks_list=[]
    for i,mask in enumerate(masks):
        mask_seg=mask['segmentation']
        mask_seg=mask_seg.astype(int)
        blocks=clip.seg_img(mask_seg,h,w,n)
        block_list=[]

        for block in blocks:
            block=block.mean()
            block_list.append(block)

        block_list=np.array(block_list)
        block_list=block_list.reshape(n,n)

        temp_block_list = np.where(block_list < maximum, 0, block_list)
        temp_block_list=np.where(temp_block_list>=maximum,i,temp_block_list)
        sam_label=sam_label+temp_block_list

        blocks_list.append(block_list)

    return blocks_list,sam_label



def note_PartOfObject_label(image,bboxes):
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
    sam.to('cuda')

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,  # 32 86  10 24  随机点,该值越大，分割结果越细
        pred_iou_thresh=0.92,
    )

    blocks=[]
    masks=[]

    print(f"bboxes:{bboxes}")
    for i in range(len(bboxes)):
        x,y,w,h,_,_=bboxes[i]
        # print(image)
        # print(f"type:{type(image)}") #numpy.ndarray
        block=image[int(y):int(y+h),int(x):int(x+w)]

        mask = mask_generator.generate(block)

        blocks.append(block)
        masks.append(mask)

        # plt.figure(figsize=(20, 20))
        # plt.subplot(1, 2, 1)
        # plt.imshow(block)
        # Segment.show_anns(mask)
        # plt.waitforbuttonpress()
    return blocks,masks


def note_edge_label(block_list,n,minimum,maximum):
    """

    :param block_list:  list n*n block
    :param n: int
    :return: edge_laebl: ndarray n*n
             b_dic:  dic  {maskID:percentage}
    """

    b_dic={}
    for i in range(n):
        for j in range(n):
            b_dic[(i,j)]={}

    for k,blocks in enumerate(block_list):

       t_block=np.array(blocks).reshape(n,n)
       index=np.where((t_block>minimum) & (t_block<maximum))
       for i,j in zip(index[0],index[1]):
         b_dic[(i,j)][k]=math.floor(t_block[i][j]*1000)/1000  #float  .3f


    edge_label=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if len(b_dic[(i,j)]) !=0:
                edge_label[i][j]=-1
    # print(edge_label)

    return edge_label,b_dic





