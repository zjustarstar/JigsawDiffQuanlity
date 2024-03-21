"""
    this part can show the image after croping, and return a list(store the n*n blocks)

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_blocks(path, n):
    """path is img_path,
        n is the size of block
    """
    image = cv2.imread(path)

    # get w,h
    height, width = image.shape[:2]

    block_h = height // n
    block_w = width // n
    blocks = seg_img(image, block_h, block_w, n)

    return blocks


def seg_img(image, h, w, n):
    """
        crop image
        return a list
        """
    block_h = h // n
    block_w = w // n

    blocks = []
    for i in range(n):
        for j in range(n):
            block = image[i * block_h: (i + 1) * block_h, j * block_w: (j + 1) * block_w]
            blocks.append(block)

    return blocks


# the follow is another  part,use check_imgwithblock can see the image

def check_imgwithblock(image, n):


    # get w,h
    height, width = image.shape[:2]

    block_h = height // n
    block_w = width // n
    blocks = seg_img(image, height, width, n)
    result_img = fix_image(blocks, block_h, block_w, height, width)
    result_img = draw_line(blocks, block_h, block_w, result_img, height, width)

    # cv2.imshow("check", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result_img

def fix_image(blocks, block_h, block_w, height, width):
    result_image = np.zeros((height, width, 4), dtype=np.uint8)
    for i, block in enumerate(blocks):
        row = i // 10
        col = i % 10
        result_image[row * block_h: (row + 1) * block_h, col * block_w: (col + 1) * block_w] = block

    return result_image


def draw_line(blocks, block_h, block_w, result, height, width):
    line_color = (0, 0, 0)

    for i, block in enumerate(blocks):
        row = i // 10
        col = i % 10
        result[row * block_h: (row + 1) * block_h, col * block_w: (col + 1) * block_w] = block
    # draw line
    for i in range(1, 10):
        x = i * block_w
        cv2.line(result, (x, 0), (x, height), line_color, 1)
    for i in range(1, 10):
        y = i * block_h
        cv2.line(result, (0, y), (width, y), line_color, 1)

    return result


def show_imgresult(path, n, patches):
    image = cv2.imread(path)

    image = cv2.resize(image, (int(image.shape[1] / 1.3), int(image.shape[0] / 1.3)))

    print(type(patches[0]))
    # get w,h
    height, width = image.shape[:2]

    block_h = height // n
    block_w = width // n
    blocks = seg_img(image, block_h, block_w, n)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.4
    font_thickness = 1

    for i, block in enumerate(blocks):
        x = i % 10 * block_w
        y = i // 10 * block_h

        line = 10
        text1 = "E: " + str(patches[i].edge)
        text_size, _ = cv2.getTextSize(text1, font, font_size, font_thickness)
        text_x = x + (block_w - text_size[0]) // 2
        text_y = y + (block_h + text_size[1] + line) // 2

        cv2.putText(image, text1, (text_x, text_y), font, font_size, (192, 192, 192), font_thickness, cv2.LINE_AA)

        text2 = "O: " + str(patches[i].object)
        text_size, _ = cv2.getTextSize(text1, font, font_size, font_thickness)
        text_x = x + (block_w - text_size[0]) // 2
        text_y = y + (block_h + text_size[1] - line) // 2

        cv2.putText(image, text2, (text_x, text_y), font, font_size, (0, 65, 255), font_thickness, cv2.LINE_AA)

    result_img = fix_image(blocks, block_h, block_w, height, width)

    result_img = draw_line(blocks, block_h, block_w, result_img, height, width)

    return result_img


def block_s_img(img,h,w,n):
    b_h=h//n
    b_w=w//n
    for i in range(1,n):
        start=(0,i*b_h)
        end=(w,i*b_h)
        cv2.line(img,start,end,(0,0,0),1)

    for i in range(1,n):
        start=(i*b_w,0)
        end=(i*b_w,h)
        cv2.line(img,start,end,(0,0,0),1)

    return img

