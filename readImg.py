import os
import cv2
import tqdm

def readImageTime(img_path):
        """

        :param img_path:  string
        :return: time_8:string 8*8
                 time_10:string 10*10
        """
        img=os.path.basename(img_path)
        name=img.split(".")
        name=name[0].split("_")
        time_8=name[1]
        time_10=name[2]

        return time_8,time_10



