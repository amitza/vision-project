import math
import os.path
import piq
import re
from itertools import chain

import torch
from tensorflow.python.ops.image_ops_impl import ssim
import cv2.cv2 as cv2
import h5py
import numpy as np

font_categories = {
    'Alex Brush': 0,
    'Open Sans': 1,
    'Sansation': 2,
    'Ubuntu Mono': 3,
    'Titillium Web': 4,
}

categories_font = {
    0: 'Alex Brush',
    1: 'Open Sans',
    2: 'Sansation',
    3: 'Ubuntu Mono',
    4: 'Titillium Web',
}


def preprocess_data(training_file: str, out_file: str, with_labels: bool, save_images: bool = False) -> [(str, int, int)]:
    db = h5py.File(training_file, 'r')
    dsets = sorted(db['data'].keys())
    data = []
    labels = []

    if save_images:
        os.mkdir('cropped_images_dir')
    chars_img_names = []
    quality = []
    for k in dsets:
        rgb = db['data'][k][...]
        charsBB = db['data'][k].attrs['charBB']
        txt = db['data'][k].attrs['txt']
        if with_labels:
            fonts = db['data'][k].attrs['font']

        img_height, img_width, img_channels = rgb.shape
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        txt = [re.split(' \n|\n |\n| ', t.decode('UTF-8').strip()) for t in txt]
        txt = list(chain(*txt))
        txt = [t for t in txt if len(t) > 0]

        c = 0
        for word_index in range(len(txt)):
            for char_index in range(len(txt[word_index])):
                chars_img_names.append([k, word_index, char_index])

                charBB = charsBB[:, :, c]
                pts1 = np.float32([[charBB[0][0], charBB[1][0]],
                                   [charBB[0][3], charBB[1][3]],
                                   [charBB[0][1], charBB[1][1]],
                                   [charBB[0][2], charBB[1][2]]])
                height = math.sqrt((charBB[0][0] - charBB[0][3]) ** 2 + (charBB[1][0] - charBB[1][3]) ** 2)
                width = math.sqrt((charBB[0][0] - charBB[0][1]) ** 2 + (charBB[1][0] - charBB[1][1]) ** 2)

                if with_labels:
                    if (height * width) <= 0:
                        err_log = 'empty file : {}\t{}\t{}\n'.format(k, txt[word_index], charBB)
                        print(err_log)
                        continue

                    if (height * width) > (img_height * img_width):
                        err_log = 'too big box : {}\t{}\t{}\n'.format(k, txt[word_index], charBB)
                        print(err_log)
                        continue

                    for i in range(2):
                        for j in range(4):
                            pos = charBB[i][j]
                            if pos < 0 or pos > rgb.shape[1 - i]:
                                err_log = f'invalid coord: ({pos})'
                                print(err_log)
                                continue

                pts2 = np.float32([[0, 0],
                                   [0, height],
                                   [width, 0],
                                   [width, height]])

                M = cv2.getPerspectiveTransform(pts1, pts2)
                cropped_image = cv2.warpPerspective(rgb, M, (int(width), int(height)))
                cropped_image = cv2.resize(cropped_image, (64, 64), interpolation=cv2.INTER_AREA)

                if save_images:
                    char_img_name = f'{k}-{word_index}-{char_index}.jpg'
                    cv2.imwrite(os.path.join('cropped_images_dir', char_img_name), cropped_image)

                data.append(cropped_image)

                if with_labels:
                    labels.append(font_categories[fonts[c].decode('UTF-8')])
                c += 1

    db.close()

    aug_data = data.copy()
    all_labels = labels.copy()

    aug_data = np.asarray(aug_data, dtype=np.uint8).reshape((len(aug_data), 64, 64, 3))
    all_labels = np.asarray(all_labels)
    np.savez(out_file, x=aug_data, y=all_labels)
    print(quality)
    return chars_img_names


def from_categorical(arr):
    y = np.argmax(arr, axis=-1)

    labels = []
    for yi in y:
        labels.append(categories_font[yi])

    return labels
