from PIL import Image
from collections import defaultdict, OrderedDict
import cv2
import torch.utils.data as data
from torchvision import datasets, transforms
import random
import pandas as pd
import numpy as np
import os
import functools

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
CATEGORIES = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]


def find_inputs(folder, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((base, abs_filename))
    return sorted(inputs, key=lambda k: k[0])


def find_targets(folder, inputs, types=IMG_EXTENSIONS):
    inputs_set = {k for k, _ in inputs}
    targets = defaultdict(dict)
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                split = base.split('-')
                if len(split) < 3:
                    continue
                if split[0] in inputs_set:
                    abs_filename = os.path.join(root, rel_filename)
                    targets[split[0]][int(split[2])] = abs_filename
    return targets


def gen_target(input_id, input_img, coords, boxes, input_mask=None):
    #FIXME on the fly gen of target patches work in progress
    h, w = input_img.shape[:2]

    for cat_idx, cat_name in enumerate(CATEGORIES):
        yx = coords[coords[:, 2] == cat_idx][:, :2]

        #FIXME bbox calcs

        timg = np.zeros([h, w])
        for y, x in yx:
            timg[y, x] += 25.

        # OpenCV guassian blur
        timg = cv2.GaussianBlur(timg, (15, 15), 3, borderType=cv2.BORDER_REFLECT_101)
        if input_mask:
            timg = cv2.bitwise_and(timg, timg, mask=input_mask)

        if True:
            print(np.min(timg), np.max(timg))
            test_sum = np.sum(timg) / 25
            ui16 = np.iinfo(np.uint16)
            test_sum = test_sum * ui16.max
            test_sum = test_sum.astype('uint16')
            test_float = test_float / ui16.max
            test_float_sum = np.sum(test_float) / 25
            print(test_sum, test_float_sum)


def gen_mask(input_img, dotted_file):
    img_dotted = cv2.imread(dotted_file)
    mask = cv2.cvtColor(img_dotted, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    img_masked = cv2.bitwise_and(input_img, input_img, mask=mask)
    return img_masked, mask


class SealionDataset(data.Dataset):
    def __init__(
            self,
            input_root,
            target_root,
            counts_file,
            coords_file='',
            tile_size=256,
            transform=None,
            target_transform=None):

        self.cat_to_idx = {cat: idx for idx, cat in enumerate(CATEGORIES)}
        counts_df = pd.read_csv(counts_file)
        counts = counts_df.to_records()

        inputs = find_inputs(input_root, types=['.npy'])
        if len(inputs) == 0:
            raise(RuntimeError("Found 0 images in : " + input_root))

        targets = find_targets(target_root, inputs)

        self.counts = counts
        self.inputs = inputs
        self.targets = targets
        self.tile_size = 256 # 256 x 256 sample/tiles
        self.transform = transform
        self.target_transform = target_transform

    @functools.lru_cache(128)
    def _load_input(self, index):
        input_id, path = self.inputs[index]
        if os.path.splitext(path)[1] == '.npy':
            img = np.load(path, mmap_mode='r')
        else:
            img = Image.open(path).convert('RGB')
        return img, input_id

    @functools.lru_cache(128)
    def _load_target(self, input_id):
        target = None
        if input_id in self.targets:
            #tp = [Image.open(self.targets[input_id][x]) for x in range(5)]
            tp = [cv2.imread(self.targets[input_id][x], -1) for x in range(5)]
            target = np.dstack(tp)
            target = target / np.iinfo(np.uint16).max
        return target

    def _random_bbox_and_transform(self, size, bounds):
        transforms.RandomHorizontalFlip
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rotate = random.random() < 0.25 if not hflip and not vflip else False
        if rotate:
            rot = random.random() * 360
        trans_x = random.randint(bounds[0][0], bounds[1[0]])
        trans_y = random.randint(bounds[0][1], bounds[1][1])
        scale = random.uniform(0.8, 1.25)
        crop = size * scale
        return None

    def _transform(self, input_img, target_arr):

        h, w = input_img.shape[:2]
        center = (w // 2, h // 2)
        angle = 0
        scale = 0.5
        hflip = False
        vflip = False
        #M = np.float32([[1, 0, x], [0, 1, y]])
        M = cv2.getRotationMatrix2D(center, angle, scale)
        if hflip:
            M[0, 0] = -M[0, 0]
        print(M)
        rotated = cv2.warpAffine(input_img, M, (w, h))
        cv2.imwrite('test.jpg', rotated)

        #input_img_bounds = np.array([[0, 0], [input_img.shape[1], input_img.shape[0]]])
        #bbox = self._random_bbox(self.tile_size, input_img_bounds)

    def __getitem__(self, index):

        input_img, input_id = self._load_input(index)
        target_arr = self._load_target(input_id)

        bb = cv2.resize(target_arr, (1024, 768))
        print(bb.shape)

        input_tile, target_tile = self._transform(input_img, target_arr)

        return input_tile, target_tile

    def __len__(self):
        return len(self.inputs)
