import cv2
import numpy as np
import pandas as pd
import argparse
import os
import sys
from collections import Counter

COLS = ['filename', 'width', 'height', 'pixels', 'aspect', 'mean', 'mean_rgb', 'std_rgb']


def extract_image_info(abs_filename, base_filename):
    info = dict()
    img = cv2.imread(abs_filename)
    info['filename'] = base_filename
    info['height'] = img.shape[0]
    info['width'] = img.shape[1]
    info['pixels'] = img.shape[0] * img.shape[1]
    info['aspect'] = float(img.shape[1]) / img.shape[0]
    bgr_mean, bgr_std = cv2.meanStdDev(img)
    info['mean_rgb'] = bgr_mean[[2, 1, 0]]
    info['std_rgb'] = bgr_std[[2, 1, 0]]
    info['mean'] = np.mean(bgr_mean)
    return info


def get_image_info(folder, types=('.jpg', '.jpeg')):
    filenames = []
    infos = []
    for root, _, files in os.walk(folder, topdown=False):
        for f in files:
            if os.path.splitext(f)[1].lower() in types:
                abs_filename = os.path.join(root, f)
                filenames.append(abs_filename)
                infos.append(extract_image_info(abs_filename, f))
    return filenames, infos


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    path = '/data/x/sealion/Train'

    filenames, infos = get_image_info(path)
    print(infos)
    info_df = pd.DataFrame.from_records(infos, columns=COLS)

    info_df.to_csv('sealion_info.csv', index=False)


if __name__ == '__main__':
    main()
