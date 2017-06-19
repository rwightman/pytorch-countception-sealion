from collections import defaultdict, OrderedDict
import cv2
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import random
import pandas as pd
import numpy as np
import os
import functools
import time
from contextlib import contextmanager
import mytransforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
CATEGORIES = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]
CATEGORY_MAP = {"adult_males": 0, "subadult_males": 1, "adult_females": 2, "juveniles": 3, "pups": 4}


@contextmanager
def measure_time(title='unknown'):
    t1 = time.clock()
    yield
    t2 = time.clock()
    print('%s: %0.2f seconds elapsed' % (title, t2-t1))


def get_crop_size(target_w, target_h, angle, scale):
    crop_w = target_w
    crop_h = target_h
    if angle:
        corners = np.array(
            [[target_w/2, -target_w/2, -target_w/2, target_w/2],
            [target_h/2, target_h/2, -target_h/2, -target_h/2]])
        s = np.sin(angle * np.pi/180)
        c = np.cos(angle * np.pi/180)
        M = np.array([[c, -s], [s, c]])
        rotated_corners = np.dot(M, corners)
        crop_w = 2 * np.max(np.abs(rotated_corners[0, :]))
        crop_h = 2 * np.max(np.abs(rotated_corners[1, :]))
    crop_w = int(np.ceil(crop_w / scale))
    crop_h = int(np.ceil(crop_h / scale))
    #print(crop_w, crop_h)
    return crop_w, crop_h


def crop_around(img, cx, cy, crop_w, crop_h):
    img_h, img_w = img.shape[:2]
    trunc_top = trunc_bottom = trunc_left = trunc_right = 0
    left = cx - crop_w//2
    if left < 0:
        trunc_left = 0 - left
        left = 0
    right = left - trunc_left + crop_w
    if right > img_w:
        trunc_right = right - img_w
        right = img_w
    top = cy - crop_h//2
    if top < 0:
        trunc_top = 0 - top
        top = 0
    bottom = top - trunc_top + crop_h
    if bottom > img_h:
        trunc_bottom = bottom - img_h
        bottom = img_h
    if trunc_left or trunc_right or trunc_top or trunc_bottom:
        #print('truncated, %d, %d, %d, %d' % (cx, cy, crop_w, crop_h))
        img_new = np.zeros((crop_h, crop_w, img.shape[2]), dtype=img.dtype)
        trunc_bottom = crop_h - trunc_bottom
        trunc_right = crop_w - trunc_right
        img_new[trunc_top:trunc_bottom, trunc_left:trunc_right] = img[top:bottom, left:right]
        return img_new
    else:
        return img[top:bottom, left:right]


def to_tensor(arr):
    assert(isinstance(arr, np.ndarray))
    t = torch.from_numpy(arr.transpose((2, 0, 1)))
    #print(t.size())
    if isinstance(t, torch.ByteTensor):
        return t.float().div(255)
    return t


def find_inputs(folder, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((int(base), abs_filename))
    if inputs:
        return zip(*sorted(inputs, key=lambda k: k[0]))
    else:
        return [], []


def find_targets(folder, input_ids, types=IMG_EXTENSIONS):
    inputs_set = set(input_ids)
    targets = defaultdict(dict)
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                split = base.split('-')
                fid = int(split[0])
                if fid in inputs_set:
                    abs_filename = os.path.join(root, rel_filename)
                    print(split)
                    if len(split) > 2:
                        targets[fid][int(split[2])] = abs_filename
                    else:
                        targets[fid] = abs_filename
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


class SampleTileIndex:
    def __init__(self, sample_index, tile_index=0):
        self.sample_index = sample_index
        self.tile_index = tile_index


class SequentialTileSampler(Sampler):
    """Samples tiles across images sequentially, always in the same order.
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)
        #self.sample_tile_map = data_source.sample_tile_map
        #FIXME not complete

    def __iter__(self):
        samples = range(self.num_samples)
        return iter(samples)
        #for i in samples:
        #    for t in self.sample_tile_map[i]:
        #        yield SampleTileIndex(i, t)

    def __len__(self):
        return self.num_samples # FIXME add up tiles


class RandomTileSampler(Sampler):
    """Oversamples random tiles from images in random order.
        Repeats the same image index multiple times in a row to sample 'repeat' times
        from the same image for big read efficiency gains.
    """
    def __init__(self, data_source, oversample=32, repeat=1):
        self.oversample = oversample//repeat * repeat
        self.repeat = repeat
        self.num_samples = len(data_source)

    def __iter__(self):
        # There are simpler/more compact ways of doing this, but why not have a somewhat
        # meaningful fake tile index?
        for to in range(self.oversample//self.repeat):
            samples = torch.randperm(self.num_samples).long()
            for image_index in samples:
                for ti in range(self.repeat):
                    tile_index = to * self.repeat + ti
                    yield SampleTileIndex(image_index, tile_index)

    def __len__(self):
        return self.num_samples * self.oversample


class SealionDataset(data.Dataset):
    def __init__(
            self,
            input_root,
            target_root='',
            counts_file='',
            coords_file='',
            processing_file='',
            train=True,
            tile_size=[256, 256],
            transform=None,
            target_transform=None):

        input_ids, input_paths = find_inputs(input_root, types=['.jpg'])
        if len(input_ids) == 0:
            raise(RuntimeError("Found 0 images in : " + input_root))
        self.input_index = input_ids
        self.data_by_id = {k: {'input': v} for k, v in zip(input_ids, input_paths)}

        self.has_targets = False
        if os.path.exists(target_root):
            targets = find_targets(target_root, input_ids, types=['.npz'])
            if len(targets):
                for k, v in targets.items():
                    d = self.data_by_id[k]
                    d['target'] = v
                self.has_targets = True
            else:
                raise (RuntimeError("Found 0 targets in : " + target_root))

        if train:
            assert self.has_targets
        self.train = train

        if counts_file:
            counts_df = pd.read_csv(counts_file).rename(columns=CATEGORY_MAP)
            counts_df.drop(['train_id'], 1, inplace=True)
            for k, v in counts_df.to_dict(orient='index').items():
                if k in self.data_by_id:
                    d = self.data_by_id[k]
                    d['counts_by_cat'] = v
                    d['count'] = sum(v.values())

        if processing_file:
            process_df = pd.read_csv(processing_file, index_col=False)
            cols = ['x_offset', 'y_offset', 'width', 'height']
            process_df[cols] = process_df[cols].astype(int)
            process_df['train_id'] = process_df.filename.map(lambda x: int(os.path.splitext(x)[0]))
            process_df.set_index(['train_id'], inplace=True)
            for k, v in process_df[cols].to_dict(orient='index').items():
                if k in self.data_by_id:
                    d = self.data_by_id[k]
                    d['process'] = v

        if coords_file:
            coords_df = pd.read_csv(coords_file, index_col=False)
            coords_df.x_coord = coords_df.x_coord.astype('int')
            coords_df.y_coord = coords_df.y_coord.astype('int')
            coords_df.category = coords_df.category.replace(CATEGORY_MAP)
            groupby_file = coords_df.groupby(['filename'])
            for file in groupby_file.indices:
                coords = groupby_file.get_group(file)
                coords = coords[['x_coord', 'y_coord', 'category']].as_matrix()
                coords.sort(axis=0)
                fid = int(os.path.splitext(file)[0])
                if fid in self.data_by_id:
                    d = self.data_by_id[fid]
                    if 'process' in d:
                        xy_offset = np.array([d['process']['x_offset'], d['process']['y_offset']])
                        coords[:, :2] = coords[:, :2] + xy_offset
                    d['coords'] = coords

        self.tile_size = tile_size
        self.dataset_mean = [0.43632373, 0.46022959, 0.4618598]
        self.dataset_std = [0.17749958, 0.16631233, 0.16272708]
        if transform is None:
            if self.train:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    mytransforms.ColorJitter(),
                    transforms.Normalize(self.dataset_mean, self.dataset_std)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.dataset_mean, self.dataset_std)
                ])
        self.target_transform = target_transform

    @functools.lru_cache(4)
    def _load_input(self, input_id):
        path = self.data_by_id[input_id]['input']
        print("Loading %s" % path)
        if os.path.splitext(path)[1] == '.npy':
            img = np.load(path)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @functools.lru_cache(4)
    def _load_target(self, input_id):
        d = self.data_by_id[input_id]
        if isinstance(d['target'], dict):
            tp = [cv2.imread(d['target'][x], -1) for x in range(5)]
            target = np.dstack(tp)
            target = target / np.iinfo(np.uint16).max
            target = target.astype(np.float32, copy=False)
        else:
            target = np.load(d['target'])['arr_0']
            target = target.astype(np.float32, copy=False)
        return target

    def _indexed_tile_center(self, input_id, index):
        d = self.data_by_id[input_id]
        cx, cy = 0, 0
        return cx, cy

    def _random_tile_center(self, input_id, w, h):
        d = self.data_by_id[input_id]
        if len(d['coords']) and random.random() < 0.4:
            # 40% of the time, randomly pick a point around an actual sealion
            cx, cy, _ = d['coords'][random.randint(0, len(d['coords']) - 1)]
            cx = cx + random.randint(-self.tile_size[0]//4, self.tile_size[0]//4)
            cy = cy + random.randint(-self.tile_size[1]//4, self.tile_size[1]//4)
        else:
            # return random center coords for specified tile size within a specified (x, y, w, h) bounding box
            tw, th = self.tile_size[0] // 2, self.tile_size[0]//2
            x_min = tw
            x_max = w - tw
            y_min = th
            y_max = h - th
            if 'process' in d:
                p = d['process']
                xo, yo = p['x_offset'], p['y_offset']
                ow, oh = p['width'], p['height']
                x_min += xo
                x_max -= w - ow - xo
                y_min += yo
                y_max -= h - oh - yo
                #print(x_min, x_max, y_min, y_max, w, h, ow, oh)
            assert x_max - x_min > 0 and y_max - y_min > 0
            cx = random.randint(x_min, x_max)
            cy = random.randint(y_min, y_max)
        return cx, cy

    def _crop_and_transform(self, cx, cy, input_img, target_arr, randomize=False):
        target_tile = None
        transform_target = False if target_arr is None else True
        if randomize:
            angle = 0.
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            do_rotate = random.random() < 0.25 if not hflip and not vflip else False
            if do_rotate:
                angle = random.random() * 360
            scale = random.uniform(0.667, 1.5)
            #print('hflip: %d, vflip: %d, angle: %f, scale: %f' % (hflip, vflip, angle, scale))
        else:
            angle = 0.
            scale = 1.
            hflip = False
            vflip = False

        crop_w, crop_h = get_crop_size(self.tile_size[0], self.tile_size[1], angle, scale)
        input_tile = crop_around(input_img, cx, cy, crop_w, crop_h)
        if transform_target:
            target_tile = crop_around(target_arr, cx, cy, crop_w, crop_h)

        # Perform tile geometry transforms if needed
        if angle or scale != 1. or hflip or vflip:
            Mtrans = np.identity(3)
            Mtrans[0, 2] = (self.tile_size[0] - crop_w) // 2
            Mtrans[1, 2] = (self.tile_size[1] - crop_h) // 2
            if hflip:
                Mtrans[0, 0] *= -1
                Mtrans[0, 2] = self.tile_size[0] - Mtrans[0, 2]
            if vflip:
                Mtrans[1, 1] *= -1
                Mtrans[1, 2] = self.tile_size[1] - Mtrans[1, 2]

            if angle or scale != 1.:
                Mrot = cv2.getRotationMatrix2D((crop_w//2, crop_h//2), angle, scale)
                Mfinal = np.dot(Mtrans, np.vstack([Mrot, [0, 0, 1]]))
            else:
                Mfinal = Mtrans

            #print(input_tile.shape)
            input_tile = cv2.warpAffine(input_tile, Mfinal[:2, :], tuple(self.tile_size))
            if transform_target:
                tt64 = target_tile.astype(np.float64)
                tt64 = cv2.warpAffine(tt64, Mfinal[:2, :], tuple(self.tile_size))
                if scale != 1.:
                    tt64 /= scale**2
                target_tile = tt64.astype(np.float32)

        return input_tile, target_tile

    def __getitem__(self, index):
        if isinstance(index, SampleTileIndex):
            tile = index.tile_index
            index = index.sample_index
        else:
            tile = 0  #FIXME sort this out

        input_id = self.input_index[index % len(self)]
        input_img = self._load_input(input_id)
        #print(input_id, index, tile)

        if self.train:
            target_arr = self._load_target(input_id)
            h, w = input_img.shape[:2]
            attempts = 32
            for i in range(attempts):
                tw, th = self.tile_size
                cx, cy = self._random_tile_center(input_id, w, h)
                input_tile, target_tile = self._crop_and_transform(cx, cy, input_img, target_arr, randomize=True)
                # check centre of chosen tile for valid pixels
                if np.any(crop_around(input_tile, tw//2, th//2, tw//4, th//4)):
                    break
            input_tile_tensor = self.transform(input_tile)
            target_tile_tensor = to_tensor(target_tile)
        else:
            target_arr = None
            if self.has_targets:
                target_arr = self._load_target(input_id)
            cx, cy = self._indexed_tile_center(input_id, index)
            input_tile, target_tile = self._crop_and_transform(cx, cy, input_img, target_arr, randomize=False)
            input_tile_tensor = self.transform(input_tile)
            if target_tile is None:
                target_tile_tensor = torch.from_numpy(np.array([input_id, tile]))
            else:
                target_tile_tensor = to_tensor(target_tile)
            #print(input_tile_tensor.size(), target_tile_tensor)

        #cv2.imwrite('test-scaled-input-%d.png' % index, input_tile)
        #cv2.imwrite('test-scaled-target-%d.png' % index, 4096*target_tile[:, :, :3])

        return input_tile_tensor, target_tile_tensor

    def __len__(self):
        return len(self.input_index)


def combine_patches(output_img, patches, patch_size, stride, agg_fn=np.mean):
    oh, ow = output_img.shape[:2]
    pw, ph = patch_size
    patch_rows = ow // pw + 1
    for i in range(0, ow):
        pi_left = ((i - pw) // stride) + 1
        pi_right = i // stride + 1
        for j in range(0, oh):
            pj_top = ((j - ph) // stride) + 1
            pj_bottom = j // stride + 1
            agg = []
            for pi in range(pi_left, pi_right):
                for pj in range(pj_top, pj_bottom):
                    px = i - pi * stride
                    py = j - pj * stride
                    agg.append(patches[pi + pj * patch_rows][px, py])
            output_img[i, j] = agg_fn(agg)