import cv2
import numpy as np
import os
import tqdm
import init
import clip
import display
import Segment_by_sam as seg
import matplotlib.pyplot as plt
import yolo_classify as yc
from CPatches import Cpatch
import readImg
import ImagePatchesClustering as IPC
from ultralytics import YOLO
from time import sleep
import json


def output(path,patches,time_8,time_10,idx,is_end=False):
    f = open(path, "a")
    if is_end==True:
        f.write("\n}")
        f.close()
        return
    if idx > 0:
        f.write(",\n")
    else:
        f.write("{")
    f.write(f"\"{img_name}\":{{\n")
    f.write(f"\t\"time_8\":{time_8},\n")
    f.write(f"\t\"time_10\":{time_10},\n")
    f.write("\t\"patch\":[")
    for i, patch in enumerate(patches):
        f.write("\t")
        json.dump(patch.attribute, f)
        if i < len(patches) - 1:
            f.write(",\n")
        # print(patch.attribute)
    f.write("]")
    f.write("}")
    f.close()

if __name__ == '__main__':
    # Hyperparameters
    data_path = 'testimg'
    puzzle_size = (10, 10) # (num_puzzle_h:int,num_puzzle_w:int)
    n,_=puzzle_size

    # threshold value
    minimum=0.2
    maximum=0.9

    # 去掉低于thre的mask
    sam_area_thre = 100*100

    model_sam = seg.load_sam_model()

    # yolo model
    weight_path = "./models/yolov8n.pt"
    yolo = YOLO(model=weight_path)

    img_list = [i for i in os.listdir(path=data_path) if i.endswith('.png')]
    non_border_patch_size = (puzzle_size[0]-2, puzzle_size[1]-2)

    save_filePath="./testimg/output_img/test.json"


    # f.truncate(0)
    for idx,img_name in tqdm.tqdm(enumerate(img_list)):

        # print(f"{idx},{img_name}",file=f)
        img_path = os.path.join(data_path,img_name)
        # img_path="./testimg/615524b941bd31b55a4d3ba5_12_23.png"
        print(img_name)

        image = cv2.imread(img_path)
        w,h,_=image.shape


        # get playing time 8*8,10*10 blocks
        time_8,time_10=readImg.readImageTime(img_path)


        # use sam.generator,  get sam_image,get masks   #################
        masks, seg_img= seg.seg_by_fb_sam(model_sam, image, sam_area_thre)

        # use masks, get label
        blocks_list,sam_label=yc.note_sam_object_label(image,masks,n,maximum)
        # print(sam_label)

        #get patch.type sam_label is object,
        edge_label,b_dic=yc.note_edge_label(blocks_list,n,minimum,maximum)

        #get similar part number
        cluster_ids=IPC.main(img_path,n)
        cluster_ids=cluster_ids.reshape(n,n)

        # print(type(edge_label))
        # print(type(sam_label))
        # print(type(b_dic))
        # print(type(blocks_list))

        # <class 'numpy.ndarray'>
        #<class 'numpy.ndarray'>
        #<class 'dict'>
        #<class 'list'>


        # edge_label,sam_label,masks,b_dic,block_list
        patches=[]
        temp_label=sam_label*cluster_ids
        for i in range(n):
            for j in range(n):
                In_num=0
                Out_num=0
                patch=Cpatch(i*n+j)

                #type
                patch.change_Type(sam_label[i][j])
                patch.change_Type(edge_label[i][j])
                #value
                patch.change_Value(b_dic[(i,j)])
                list_In=[]
                list_Out=[]
                #similar
                if temp_label[i][j]!=0:
                    In_num=np.where(temp_label==temp_label[i][j])
                    if type(In_num)==int:
                        list_In=[]
                    else:
                        list_In=[int(i*10+j) for i,j in zip(In_num[0],In_num[1]) ]
                Out_num=np.where(cluster_ids==cluster_ids[i][j])
                if type(Out_num)==int:
                    list_Out=[]
                else:
                    list_Out=[int(i*10+j) for i,j in zip(Out_num[0],Out_num[1]) if i*n+j not in list_In]

                patch.change_SimPatchInObj(list_In)
                patch.change_SimPatchOutObj(list_Out)

                patches.append(patch)

        #show the attribute
        output(save_filePath,patches,time_8,time_10,idx)

        #next classify the object with yolo
        class_label_list,bboxes=init.init(model_sam,yolo,img_path,n,maximum,minimum)

        if type(bboxes) == bool:
            y_image=image.copy()
        else:
            y_image = image.copy()
            y_image=display.plot_bboxes(y_image,bboxes,score=False)

        #similar part image
        sim_image=image.copy()
        sim_image = display.add_patch_line(sim_image, n, linecolor=(255, 0, 0))
        sim_image = display.add_patch_line(sim_image, n, linecolor=(255, 0, 0))
        sim_image = display.add_patch_class(sim_image,n,cluster_ids,text_color=(255,255,255))

        # 给image增加文字和标示
        seg_img = display.add_patch_line(seg_img, n, linecolor=(255, 0, 0))
        image = display.add_patch_line(image, n, linecolor=(255,0,0))
        image = display.add_patch_ID(image, n, text_color=(255,255,255))


        # # 显示
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        axs[0, 0].set_title("origin img")
        axs[0, 0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title("sam_img")
        axs[0, 1].imshow(seg_img)
        axs[1, 0].set_title("y_img")
        axs[1, 0].imshow(y_image)
        axs[1, 1].set_title("sim_img")
        axs[1, 1].imshow(cv2.cvtColor(sim_image,cv2.COLOR_BGR2RGB))
        # plt.savefig(f"./testimg/output_img/{img_name}")
        # plt.show()
        # print("next")
        # input("next:")
    output(save_filePath,is_end=True)
    # print(sam_label)
    # print(image.shape)   #(2048,2048,3)
    # s_image.shape : (2048,2048,4)

    #     blur_img=cv2.blur(cv2.imread(img_path),(5,5))
    # # n=10
    # # 每加载一张图，就需要处理一张图
    #     patches,boxes=init.init(img_path,n)
    #
    #     origin_img=cv2.imread(img_path)
    #     y_img=origin_img.copy()
    #     result_img= clip.show_imgresult(img_path,n,patches)
    #     seg_img=Segment.seg_by_fb_sam(img_path)
    #
    #     fig,axs=plt.subplots(2,2,figsize=(12,12))
    #
    #     axs[0, 0].set_title("origin")
    #     axs[0, 0].imshow(cv2.cvtColor(origin_img,cv2.COLOR_BGR2RGB))
    #     axs[0, 1].set_title("y_img")
    #     for box in boxes:
    #
    #         x,y,w,h,_,_=map(int,box)
    #         cv2.rectangle(y_img,(x,y),(x+w,y+h),(0,255,0),2)
    #
    #     axs[0, 1].imshow(cv2.cvtColor(y_img,cv2.COLOR_BGR2RGB))
    #     axs[1, 0].set_title("result")
    #     axs[1, 0].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    #     axs[1, 1].set_title("seg")
    #     axs[1, 1].imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
    #
    #     plt.suptitle(f'{time_8}_{time_10}')
    #     # plt.tight_layout()
    #
    #     plt.show()
    #     plt.waitforbuttonpress()
