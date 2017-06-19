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
CATEGORIES = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]
CATEGORY_MAP = {"adult_males": 0, "subadult_males": 1, "adult_females": 2, "juveniles": 3, "pups": 4}


def get_outdir(parent_dir, child_dir=''):
    outdir = os.path.join(parent_dir, child_dir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def find_images(folder, types=('.jpg', '.jpeg')):
    results = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            if os.path.splitext(rel_filename)[1].lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                results.append((rel_filename, abs_filename))
    return results


class Process(object):
    def __init__(self, root_path, counts_filename, coords_filename, dest_name='Train-processed'):

        self.input_path = os.path.join(root_path, 'Train')
        self.dotted_path = os.path.join(root_path, 'TrainDotted')
        self.output_path_inputs = get_outdir(os.path.join(root_path, dest_name, 'inputs'))
        self.output_path_targets = get_outdir(os.path.join(root_path, dest_name, 'targets'))
        self.counts_df = pd.read_csv(os.path.join(root_path, counts_filename), index_col=0)
        self.coords_df = pd.read_csv(os.path.join(root_path, coords_filename), index_col=False)
        self.coords_df.x_coord = self.coords_df.x_coord.astype('int')
        self.coords_df.y_coord = self.coords_df.y_coord.astype('int')
        self.coords_df.category = self.coords_df.category.replace(CATEGORY_MAP)
        self.coords_by_file = self.coords_df.groupby('filename')
        self.bsize = 256
        self.reflect = False

    def _process_file(self, frel, fabs, results, stats=None):
        print('Processing %s...' % frel)
        basename = os.path.splitext(frel)[0]
        fid = int(basename)

        if frel not in self.coords_by_file.groups:
            print('Frame %s counts not found, skipping.' % frel)
            return

        img = cv2.imread(fabs)
        h, w = img.shape[:2]
        hb = int(math.ceil((h + self.bsize)/self.bsize) * self.bsize)
        wb = int(math.ceil((w + self.bsize)/self.bsize) * self.bsize)
        print(wb, hb)
        y_diff = hb - h
        x_diff = wb - w
        y_offset = y_diff // 2
        x_offset = x_diff // 2
        print(x_offset, y_offset)

        dotted_file = os.path.join(self.dotted_path, frel)
        if os.path.exists(dotted_file):
            img_dotted = cv2.imread(dotted_file)
            if img_dotted.shape[:2] != img.shape[:2]:
                print("Dotted image size doesn't  match train for %s, skipping..." % frel)
                return
            mask = cv2.cvtColor(img_dotted, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
            img = cv2.bitwise_and(img, img, mask=mask)
            # scale up the mask for targets
            mask = cv2.copyMakeBorder(
                mask, y_offset, y_diff-y_offset, x_offset, x_diff-x_offset, cv2.BORDER_CONSTANT, (0, 0, 0))
        else:
            print("No matching dotted file exists for %s, skipping..." % frel)
            return

        result = dict()
        result['filename'] = frel
        result['height'] = h
        result['width'] = w
        result['buffer_height'] = hb
        result['buffer_width'] = wb
        result['x_offset'] = x_offset
        result['y_offset'] = y_offset

        mean, std = cv2.meanStdDev(img, mask=mask)
        mean = mean[::-1].squeeze()/255
        std = std[::-1].squeeze()/255
        print('Mean, std: ', mean, std)
        result['mean'] = mean
        result['std'] = std
        if stats is not None:
            stats.append(np.array([mean, std]))
            if len(stats) % 10 == 0:
                print("Current avg mean, std:")
                statss = np.array(stats)
                print(np.mean(statss, axis=0))

        if self.reflect:
            border = cv2.BORDER_REFLECT_101
            value = None
        else:
            border = cv2.BORDER_CONSTANT
            value = (0, 0, 0)

        img = cv2.copyMakeBorder(img, y_offset, y_diff-y_offset, x_offset, x_diff-x_offset, border, value)
        cv2.imwrite(os.path.join(self.output_path_inputs, frel), img)
        #cv2.imwrite(os.path.join(self.output_path_inputs, basename + '.png'), img)

        print(self.counts_df.ix[fid])
        yxc = self.coords_by_file.get_group(frel).as_matrix(columns=['y_coord', 'x_coord', 'category'])

        targets = []
        write_scaled_pngs = False
        for cat_idx, cat_name in enumerate(CATEGORIES):
            yx = yxc[yxc[:, 2] == cat_idx][:, :2]
            yx += [y_offset, x_offset]

            gimg = np.zeros([hb, wb])
            for y, x in yx:
                gimg[y, x] += 1024.

            # OpenCV gaussian blur
            target_img = cv2.GaussianBlur(gimg, (19, 19), 3, borderType=cv2.BORDER_REFLECT_101)
            target_img = cv2.bitwise_and(target_img, target_img, mask=mask)
            print("Min/max: ", np.min(target_img), np.max(target_img))

            # Scipy, scipy.ndimage.filters.gaussian_filter
            #blah2 = gaussian_filter(gimg, 3)
            #if mask_used:
            #    blah2 = cv2.bitwise_and(blah2, blah2, mask=mask)
            #blah2_sum = np.sum(blah2)/10
            #blah2 = blah2 * 255
            #cv2.imwrite(os.path.join(self.output_path, basename + '-gn.jpg'), blah2)

            targets.append(target_img.astype(np.float32))
            test_sum = np.sum(target_img) / 1024
            if write_scaled_pngs:
                INT_SCALE = np.iinfo(np.uint16).max / 32
                target_img_uint16 = target_img * INT_SCALE
                target_img_uint16 = target_img_uint16.astype('uint16')

                target_path = os.path.join(self.output_path_targets, basename + '-target-%d.png' % cat_idx)
                cv2.imwrite(target_path, target_img_uint16)

                # read back and verify
                target_img_float = cv2.imread(target_path, -1) / INT_SCALE
                target_img_float_sum = np.sum(target_img_float) / 1024

                print('Counts for class %d:' % cat_idx, test_sum, target_img_float_sum, len(yx))
                assert(np.isclose(test_sum, target_img_float_sum, rtol=.001, atol=.001))

        target_stacked = np.dstack(targets)
        target_path = os.path.join(self.output_path_targets, basename + '-target.npz')
        np.savez_compressed(target_path, target_stacked)

        results.append(result)

    def __call__(self):
        if not os.path.isdir(self.input_path):
            print('Error: Folder %s does not exist.' % self.input_path)
            return []
        inputs = find_images(self.input_path)
        if not inputs:
            print('Error: No inputs found at %s.' % self.input_path)
            return []
        results = []
        stats = []
        for frel, fabs in inputs:
            self._process_file(frel, fabs, results, stats)
        stats = np.array(stats)
        print('Dataset mean, std: ', np.mean(stats, axis=0))
        return results


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    root_path = '/data/x/sealion/'
    dest_name = 'Train_processed'
    counts_filename = 'Train/train.csv'
    coords_filename = 'Train/correct_coords.csv'

    process = Process(root_path, counts_filename, coords_filename, dest_name=dest_name)
    results = process()

    df = pd.DataFrame.from_records(results, columns=COLS)
    df.to_csv(os.path.join(root_path, dest_name, 'processed.csv'), index=False)

if __name__ == '__main__':
    main()