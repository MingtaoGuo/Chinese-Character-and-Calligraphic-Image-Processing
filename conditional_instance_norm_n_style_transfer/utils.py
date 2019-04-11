from PIL import Image
import numpy as np
import scipy.misc as misc
import os

def random_batch(path, batch_size, shape):
    filenames = os.listdir(path)
    rand_samples = np.random.randint(0, filenames.__len__(), [batch_size])
    batch = np.zeros([batch_size, shape[0], shape[1], shape[2]])
    c = 0
    for idx in rand_samples:
        try:
            batch[c, :, :, :] = misc.imresize(crop(np.array(Image.open(path + filenames[idx]))), [shape[0], shape[1]])
        except:
            img = crop(np.array(Image.open(path + filenames[0])))
            batch[c, :, :, :] = misc.imresize(img, [shape[0], shape[1]])
        c += 1
    return batch

def random_select_style(path, batch_size, shape, c_nums):
    filenames = os.listdir(path)
    rand_sample = np.random.randint(0, filenames.__len__())
    img = misc.imresize(crop(np.array(Image.open(path + str(rand_sample + 1) + ".png"))), [shape[0], shape[1]])
    batch = np.zeros([batch_size, shape[0], shape[1], shape[2]])
    y = np.zeros([1, c_nums])
    y[0, rand_sample] = 1
    for i in range(batch_size):
        batch[i, :, :, :] = img[:, :, :3]
    return batch, y

def crop(img):
    h = img.shape[0]
    w = img.shape[1]
    if h < w:
        x = 0
        y = np.random.randint(0, w - h + 1)
        length = h
    elif h > w:
        x = np.random.randint(0, h - w + 1)
        y = 0
        length = w
    else:
        x = 0
        y = 0
        length = h
    return img[x:x+length, y:y+length, :]