import cv2
import numpy as np
import pandas as pd
import argparse
import os
import sys
import math
import copyreg
import types
import itertools
from pathos import multiprocessing

from collections import Counter
from scipy.stats import kde
from scipy.ndimage import gaussian_filter


COLS = ['filename', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'mean', 'std']
CATEGORIES = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]
CATEGORY_MAP = {"adult_males": 0, "subadult_males": 1, "adult_females": 2, "juveniles": 3, "pups": 4}


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)


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
    def __init__(
            self,
            root_path,
            src_folder='Train',
            metadata_folder='Train',
            dst_folder='Train-processed',
            padding_size=256,
            calc_stats=True):

        self.write_inputs = True if dst_folder else False
        self.generate_targets = False
        self.input_path = os.path.join(root_path, src_folder)

        self.dotted_path = os.path.join(root_path, src_folder + 'Dotted')
        if not os.path.exists(self.dotted_path):
            self.dotted_path = ''
            print("No dotted annotation for specified input source path")

        counts_path = os.path.join(root_path, metadata_folder, 'train.csv')
        if os.path.isfile(counts_path):
            self.counts_df = pd.read_csv(counts_path, index_col=0)
        else:
            self.counts_df = pd.DataFrame()
            print("No counts metadata available at %s" % counts_path)

        coords_path = os.path.join(root_path, metadata_folder, 'correct_coords.csv')
        if os.path.isfile(coords_path):
            self.coords_df = pd.read_csv(coords_path, index_col=False)
            self.coords_df.x_coord = self.coords_df.x_coord.astype('int')
            self.coords_df.y_coord = self.coords_df.y_coord.astype('int')
            self.coords_df.category = self.coords_df.category.replace(CATEGORY_MAP)
            self.coords_by_file = self.coords_df.groupby('filename')
        else:
            self.coords_df = pd.DataFrame()
            print("No coordinates metadata available at %s, not generating targets" % coords_path)
            self.generate_targets = False

        if self.write_inputs:
            if self.generate_targets:
                self.output_path_inputs = get_outdir(os.path.join(root_path, dst_folder, 'inputs'))
                self.output_path_targets = get_outdir(os.path.join(root_path, dst_folder, 'targets'))
            else:
                self.output_path_inputs = get_outdir(os.path.join(root_path, dst_folder))
                self.output_path_targets = ''
        else:
            self.output_path_inputs = ''
            self.output_path_targets = ''

        self.padding_size = padding_size
        self.border_reflect = False
        self.calc_stats = calc_stats
        self.verify_targets = False
        self.write_scaled_pngs = False

    def _process_file(self, frel, fabs, results, stats=None):
        print('Processing %s...' % frel)
        basename = os.path.splitext(frel)[0]
        fid = int(basename)

        if len(self.coords_df) and frel not in self.coords_by_file.groups:
            print('Frame %s counts/coords not found, skipping.' % frel)
            return

        img = cv2.imread(fabs)
        h, w = img.shape[:2]
        if self.padding_size:
            wb = int(math.ceil((w + self.padding_size) / self.padding_size) * self.padding_size)
            hb = int(math.ceil((h + self.padding_size) / self.padding_size) * self.padding_size)
        else:
            wb = w
            hb = h
        x_diff = wb - w
        y_diff = hb - h
        x_min = x_diff // 2
        y_min = y_diff // 2
        x_max = x_min + w
        y_max = y_min + h

        if self.dotted_path:
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
                    mask, y_min, y_diff-y_min, x_min, x_diff-x_min, cv2.BORDER_CONSTANT, (0, 0, 0))
            else:
                print("No matching dotted file exists for %s, skipping..." % frel)
                return
        else:
            mask = None

        result = dict()
        result['id'] = fid
        result['filename'] = frel
        result['height'] = hb
        result['width'] = wb
        result['xmin'] = x_min
        result['ymin'] = y_min
        result['xmax'] = x_max
        result['ymax'] = y_max

        if self.calc_stats:
            mean, std = cv2.meanStdDev(img, mask=mask)
            mean = mean[::-1].squeeze() / 255
            std = std[::-1].squeeze() / 255
            print('Mean, std: ', mean, std)
            result['mean'] = list(mean)
            result['std'] = list(std)
            if stats is not None:
                stats.append(np.array([mean, std]))
                if len(stats) % 10 == 0:
                    print("Current avg mean, std:")
                    statss = np.array(stats)
                    print(np.mean(statss, axis=0))

        if self.write_inputs:
            if self.padding_size:
                if self.border_reflect:
                    border = cv2.BORDER_REFLECT_101
                    value = None
                else:
                    border = cv2.BORDER_CONSTANT
                    value = (0, 0, 0)
                img = cv2.copyMakeBorder(img, y_min, y_diff-y_min, x_min, x_diff-x_min, border, value)
            cv2.imwrite(os.path.join(self.output_path_inputs, frel), img)

        if self.generate_targets:
            self._generate_target(fid, frel, y_min, x_min, wb, hb, mask)

        results.append(result)

    def _generate_target(self, fid, frel, y_min, x_min, width, height, mask):
        print(self.counts_df.ix[fid])
        yxc = self.coords_by_file.get_group(frel).as_matrix(columns=['y_coord', 'x_coord', 'category'])
        targets = []
        for cat_idx, cat_name in enumerate(CATEGORIES):
            yx = yxc[yxc[:, 2] == cat_idx][:, :2]
            yx += [y_min, x_min]

            gauss_img = np.zeros([height, width])
            for y, x in yx:
                gauss_img[y, x] += 1024.

            # OpenCV gaussian blur
            target_img = cv2.GaussianBlur(gauss_img, (19, 19), 3, borderType=cv2.BORDER_REFLECT_101)
            target_img = cv2.bitwise_and(target_img, target_img, mask=mask)
            print("Min/max: ", np.min(target_img), np.max(target_img))

            # Scipy, scipy.ndimage.filters.gaussian_filter
            #gauss_img = gaussian_filter(gauss_img, 3)
            #if mask_used:
            #    blah2 = cv2.bitwise_and(blah2, blah2, mask=mask)
            #gauss_img = np.sum(blah2)/1024
            #gauss_img = gauss_img * 255

            targets.append(target_img.astype(np.float32))

            # Verification
            if self.verify_targets:
                # Note sometimes masks cut out parts of density map that contribute to counts and this fails
                test_sum = np.sum(target_img) / 1024
                print('Counts for class %d:' % cat_idx, test_sum, len(yx))
                assert np.isclose(test_sum, float(len(yx)), atol=.001)

            if self.write_scaled_pngs:
                INT_SCALE = np.iinfo(np.uint16).max / 32
                target_img_uint16 = target_img * INT_SCALE
                target_img_uint16 = target_img_uint16.astype('uint16')
                target_path = os.path.join(self.output_path_targets, '%d-target-%d.png' % (fid, cat_idx))
                cv2.imwrite(target_path, target_img_uint16)

        target_stacked = np.dstack(targets)
        target_path = os.path.join(self.output_path_targets, '%d-target.npz' % fid)
        np.savez_compressed(target_path, target_stacked)

    def _process_files(self, inputs):
        results = []
        stats = []
        for frel, fabs in inputs:
            self._process_file(frel, fabs, results, stats)
        return results, stats

    def __call__(self, num_processes=1):
        if not os.path.isdir(self.input_path):
            print('Error: Folder %s does not exist.' % self.input_path)
            return []
        inputs = find_images(self.input_path)
        if not inputs:
            print('Error: No inputs found at %s.' % self.input_path)
            return []
        results = []
        stats = []
        if num_processes > 1:
            input_slices = [x.tolist() for x in np.array_split(inputs, num_processes)]
            pool = multiprocessing.Pool(num_processes)
            for m in pool.map(self._process_files, input_slices):
                results += m[0]
                stats += m[1]
            results.sort(key=lambda k: k['id'])
        else:
            results, stats = self._process_files(inputs)
        stats = np.array(stats)
        print('Dataset mean, std: ', np.mean(stats, axis=0))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    args = parser.parse_args()

    root_path = args.data
    src_folder = 'Test'
    if 'Test' not in src_folder:
        dst_folder = src_folder + '-processed'
        padding_size = 256
    else:
        dst_folder = ''
        padding_size = 0
    metadata_folder = src_folder
    process = Process(
        root_path,
        src_folder=src_folder,
        metadata_folder=metadata_folder,
        dst_folder=dst_folder,
        padding_size=padding_size,
        calc_stats=True)
    results = process(4)

    df = pd.DataFrame.from_records(results, columns=COLS)
    df.to_csv(
        os.path.join(root_path, dst_folder if dst_folder else src_folder, 'processed.csv'),
        index=False)

if __name__ == '__main__':
    main()