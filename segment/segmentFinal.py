import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc as misc
import os



def find_row(img):
    row_locs = []
    locs = np.zeros([2])
    y = np.sum(img, axis=1, dtype=np.int32) - 10
    y = np.clip(y, 0, 1000)#将值裁剪到0-1000之间，更容易找到边界点
    for i in range(y.shape[0]-1):
        if y[i] == 0 and y[i + 1] > 0:
            locs[0] = i + 1
        if y[i] > 0 and y[i + 1] == 0:
            locs[1] = i + 1
            row_locs.append(locs * 1.0)
    length = row_locs.__len__()
    i = 0
    while i < length:
        if row_locs[i][1] - row_locs[i][0] < 30:
            row_locs.pop(i)
            length = row_locs.__len__()
            i -= 1
        i += 1

    return row_locs

def find_col(img):
    col_locs = []
    locs = np.zeros([2])
    y = np.sum(img, axis=0, dtype=np.int32) - 10
    y = np.clip(y, 0, 1000)
    for i in range(y.shape[0]-1):
        if y[i] == 0 and y[i + 1] > 0:
            locs[0] = i + 1
        if y[i] > 0 and y[i + 1] == 0:
            locs[1] = i + 1
            col_locs.append(locs * 1.0)
    length = col_locs.__len__()
    i = 0
    while i < length:
        if col_locs[i][1] - col_locs[i][0] < 30:
            col_locs.pop(i)
            length = col_locs.__len__()
            i -= 1
        i += 1
    return col_locs

def coarse_segment(img):
    row_locs = find_row(img)
    col_locs = find_col(img)
    coarse_char_locs = []
    for row_loc in row_locs:
        for col_loc in col_locs:
            coarse_char_locs.append([int(row_loc[0]), int(row_loc[1]), int(col_loc[0]), int(col_loc[1])])
    return coarse_char_locs

def align_char(char_img, target_h, target_w):
    canvas = np.ones([target_h, target_w], dtype=np.int32) * 255
    img_h, img_w = char_img.shape[0], char_img.shape[1]
    if img_h > img_w:
        new_h = target_h
        new_w = np.int32(img_w * target_h / img_h)
        char_img = misc.imresize(char_img, [new_h, new_w])
        mid_w = target_w // 2
        start = mid_w - new_w // 2
        end = start + new_w
        canvas[:, start:end] = char_img
    if img_h < img_w:
        new_w = target_w
        new_h = np.int32(img_h * target_w / img_w)
        char_img = misc.imresize(char_img, [new_h, new_w])
        mid_h = target_h // 2
        start = mid_h - new_h // 2
        end = start + new_h
        canvas[start:end, :] = char_img
    if img_h == img_w:
        canvas = misc.imresize(char_img, [target_h, target_w])
    return canvas


def generate_char_dataset(source_path, target_path):
    source_lists = os.listdir(source_path)
    c = 0
    for s in source_lists:
        img = np.array(Image.open(source_path + s).convert("L"))
        temp = 255 - img
        temp[temp < 100] = 0#消除无法观测到的非0点
        coarse_char_locs = coarse_segment(temp)
        for loc in coarse_char_locs:
            single_char = img[int(loc[0]):int(loc[1]), int(loc[2]):int(loc[3])]
            if np.mean(single_char) > 250:
                continue
            Image.fromarray(np.uint8(align_char(single_char, 128, 128))).save(target_path + str(c) + ".jpg")
            c += 1


if __name__ == "__main__":
    filename = "19 思源宋体 CN Light/"
    source_path = "C:/Users/gmt/Desktop/字体多风格迁移/jpg-dataset/" + filename
    target_path = "C:/Users/gmt/Desktop/字体多风格迁移/single-dataset/" + filename
    os.mkdir(target_path)
    generate_char_dataset(source_path, target_path)
