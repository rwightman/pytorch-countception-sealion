import cv2
import numpy as np
import pandas as pd
import argparse
import os
import sys
import math
from collections import Counter
from scipy.stats import kde
from scipy.ndimage import gaussian_filter


COLS = ['filename', 'width', 'height', 'buffer_width', 'buffer_height', 'x_offset', 'y_offset']

category_map = {"adult_males": 0, "subadult_males": 1, "adult_females": 2, "juveniles": 3, "pups": 4}


def get_outdir(parent_dir, child_dir=''):
    outdir = os.path.join(parent_dir, child_dir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def process_images(folder, fn, types=('.jpg', '.jpeg')):
    results = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            if os.path.splitext(rel_filename)[1].lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                results.append(fn(abs_filename, rel_filename))
    return results


class Process(object):
    def __init__(self, input_path, output_path, counts_filename, coords_filename):
        self.output_path = get_outdir(output_path)
        self.counts_df = pd.read_csv(os.path.join(input_path, counts_filename), index_col=0)
        self.coords_df = pd.read_csv(os.path.join(input_path, coords_filename), index_col=False)
        self.coords_df.x_coord = self.coords_df.x_coord.astype('int')
        self.coords_df.y_coord = self.coords_df.y_coord.astype('int')
        self.coords_df.category = self.coords_df.category.replace(category_map)
        self.coords_by_file = self.coords_df.groupby('filename')
        self.bsize = 256
        self.reflect = False


    def __call__(self, fabs, frel):
        print('Processing %s...' % frel)
        basename = os.path.splitext(frel)[0]
        result = dict()

        if frel not in self.coords_by_file.groups:
            print('Frame %s counts not found, skipping.' % frel)
            return result

        img = cv2.imread(fabs)
        h, w = img.shape[:2]

        dotted_file = os.path.join("/data/x/sealion/TrainDotted/" + frel)
        mask_used = False
        if os.path.exists(dotted_file):
            img_dotted = cv2.imread(dotted_file)
            mask = cv2.cvtColor(img_dotted, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            img = cv2.bitwise_and(img, img, mask=mask)
            mask_used = True

        result['filename'] = frel
        result['height'] = h
        result['width'] = w

        hb = math.ceil((h + self.bsize)/self.bsize) * self.bsize
        wb = math.ceil((w + self.bsize)/self.bsize) * self.bsize
        print(wb, hb)

        y_diff = hb - h
        x_diff = wb - w
        y_offset = y_diff//2
        x_offset = x_diff // 2
        print(x_offset, y_offset)

        result['buffer_height'] = hb
        result['buffer_width'] = wb
        result['x_offset'] = x_offset
        result['y_offset'] = y_offset

        if self.reflect:
            border = cv2.BORDER_REFLECT_101
            value = None
        else:
            border = cv2.BORDER_CONSTANT
            value = (0, 0, 0)

        img2 = cv2.copyMakeBorder(img, y_offset, y_diff-y_offset, x_offset, x_diff-x_offset, border, value)
        cv2.imwrite(os.path.join(self.output_path, frel), img2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        print(img2.dtype)



        print(img2.shape)
        np.save(os.path.join(self.output_path, basename), img2, allow_pickle=False)


        xyc = self.coords_by_file.get_group(frel).as_matrix(columns=['y_coord', 'x_coord', 'category'])

        # KDE method
        #if xy.T.shape[1] > 2:
            #k = kde.gaussian_kde(xy.T)
            #xg, yg = np.mgrid[0:w, 0:h]
            #zg = k(np.vstack([xg.flatten(), yg.flatten()]))
            #cv2.imwrite(os.path.join(self.output_path, basename + '-gk.jpg'), zg)

        for cat in category_map:
            val = category_map[cat]
            xy = xyc[xyc[:, 2] == val][:, :2]
            print(val, xy.shape)

            gimg = np.zeros([h, w])
            for x, y in xy:
                gimg[x, y] += 25.

            # OpenCV guassian blur
            blah = cv2.GaussianBlur(gimg, (11, 11), 3, borderType=cv2.BORDER_REFLECT_101)
            if mask_used:
                blah = cv2.bitwise_and(blah, blah, mask=mask)
            blah_sum = np.sum(blah)/25
            print(np.min(blah), np.max(blah))

            ui16 = np.iinfo(np.uint16)
            blah = blah * ui16.max
            blah = blah.astype('uint16')
            blah_float = blah/ui16.max
            blah_sum_trunc = np.sum(blah_float) / 25

            cv2.imwrite(os.path.join(self.output_path, basename + '-target-%d.png' % val), blah)

            # Scipy, scipy.ndimage.filters.gaussian_filter
            #blah2 = gaussian_filter(gimg, 3)
            #if mask_used:
            #    blah2 = cv2.bitwise_and(blah2, blah2, mask=mask)
            #blah2_sum = np.sum(blah2)/10
            #blah2 = blah2 * 255
            #cv2.imwrite(os.path.join(self.output_path, basename + '-gn.jpg'), blah2)

            print('Counts for class %d:' % val, blah_sum, blah_sum_trunc)

        return result


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    input_path = '/data/x/sealion/Train/'
    output_path = '/data/x/sealion/Train-processed/'
    counts_filename = 'train.csv'
    coords_filename = 'correct_coords.csv'

    process = Process(input_path, output_path, counts_filename, coords_filename)
    results = process_images(input_path, fn=process)

    df = pd.DataFrame.from_records(results, columns=COLS)
    df.to_csv(os.path.join(input_path, 'processed', 'info.csv'), index=False)

if __name__ == '__main__':
    main()