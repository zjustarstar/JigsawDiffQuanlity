import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import display
import torch
import yolo_classify as yc
import CPatches
import clip





def init(sam,yolo,img_path,n,maximum,minimum):
    """

    :param img_path: string
    :param n: int
    :return: patches
    """

    image,masks,boxes,results,label_list=init_model(sam,yolo,img_path,n)

    if type(masks) == bool:
        return False,False
    label_lists=com_label(image,n,masks,label_list,maximum,minimum)
    # patches=init_patch(label_lists,edge_label_list)

    # return patches,boxes

    return label_lists,boxes

def init_model(sam,yolo,img_path,n):
    """
    this part is a init,include sam,yolo,image

    :param img_path:string
    :return mask: array{:is a bool matrix}
    """


    # init image
    image = cv2.imread(img_path)
    image=cv2.blur(image,(5,5))
    # get the label,results
    label_list,results=yc.note_object_labels(yolo,image,n)

    # get boxes
    image_bbox=image.copy()
    boxes=np.array(results[0].to('cpu').boxes.data)
    if len(boxes) == 0:
        return image, False,False,False,False

    #细分object内部的object
    # P_blocks,P_masks = yc.note_PartOfObject_label(image,boxes)

    # display.plot_bboxes(image_bbox,boxes,score=False) #if want see the photo after using yolo,use this

    # print(f"boxes.shape:{boxes.shape}")
    # mistake


    # we use predictor ,not generator,generator will detect all object,and it might damage a complete object
    # so we choose predictor ,it needs some tip then start work

    # init mask_predictor
    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image)

    ######    the following is for multiple targets

    input_boxes=torch.tensor(boxes[:,:-2],device=mask_predictor.device)

    transformed_boxes = mask_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

    masks, _, _ = mask_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # get the image's masks
    masks=torch.squeeze(masks, 1) #  a bool matrix ,each pixel has a bool

    return image,masks,boxes,results,label_list

def com_label(img,n,masks,label_list,maximum,minimum):
    # 目前已经有：block，类型矩阵，像素点矩阵
    # 目标： 10*10的拼图，每一块patch都有属性，能够区分有无object，edge，score还没到
    # 方法，因为已经有一个10*10的类型矩阵，可以将像素点矩阵转换为[0,1]矩阵，再与类型矩阵比较，如果为1且有类型值，将类型值赋值给新的矩阵，这样100块完成后，便利，给patches修改值
    """

    :param img_path: string
    :param n: int
    :param masks: array
    :return: label_lists: list
    """



    label_lists= label_list


    h,w,_=img.shape


    temp_blocks=np.zeros((n,n))

    edge_label_list = []
    # deal with the masks
    for mask in masks:
        mask=mask.type(torch.float32)
        mask=mask.to('cpu').numpy()

        # here use clip.seg_img,then it will return a matrix(type is list)
        blocks=clip.seg_img(mask,h,w,n)

        block_list=[]

        for block in blocks:
            block=block.mean()  #bool->float32  (n*n)*1
            block_list.append(block)
        block_list=np.array(block_list)
        block_list=block_list.reshape(n,n)

        # edge_label= yc.note_edge_label(np.where(block_list>0.2,1,block_list),n) #edge label
        # edge_label_list.append((edge_label))

        # my idea is : the precision  yolo < sam,  some part in label is redundant,so we need updata
        temp_blocks=temp_blocks + np.where(block_list>minimum,1,block_list)

    temp_blocks=np.where(temp_blocks>0,1,temp_blocks)


    for i,label in enumerate(label_lists):
        label=label*temp_blocks
        label_lists[i]=label


    return label_lists

def init_patch(label_lists,edge_label_list):
    #init patch,
    """

    :param label_lists: list(array(list))
    :return: patches: list(CPatches)
    """
    # print(edge_label_list)
    patches=[]
    for i,label in enumerate(label_lists):

        for val in label:
            for j in range(len(val)):
                patch=CPatches.Cpatch()
                patch.is_object(val[j])
                patches.append(patch)
    for k,edge_label in enumerate(edge_label_list):
        for i,list in enumerate(edge_label):
            # print(list)
            for j in range(len(list)):
                patches[i*len(list)+j].is_edge(int(list[j]))
                # print(patches[i*len(list)+j].edge)
                # print(list[j])
        #
    # list=[]
    # for i in range(100):
    #     list.append(patches[i].edge)
    # print(np.array(list).reshape((10,10)))
    return patches

def classify_bin(patches):
    bin={
        "0_5":{},
        "5_10": {},
        "10_15": {},
        "15_20": {}
    }
    temp_dict={}
    for i in range(patches):
        if patches[i].attribute["type"]=="Object":
            print("111")
