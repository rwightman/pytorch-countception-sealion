from collections import defaultdict, OrderedDict
import cv2
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
from PIL import Image
import torchvision.utils as tvutils
import random
import pandas as pd
import numpy as np
import os
import functools
import time
import mytransforms
import utils
import utils_cython

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
CATEGORIES = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]
CATEGORY_MAP = {"adult_males": 0, "subadult_males": 1, "adult_females": 2, "juveniles": 3, "pups": 4}


def to_tensor(arr):
    assert(isinstance(arr, np.ndarray))
    t = torch.from_numpy(arr.transpose((2, 0, 1)))
    #print(t.size())
    if isinstance(t, torch.ByteTensor):
        return t.float().div(255)
    return t


def find_inputs(folder, types=IMG_EXTENSIONS, extract_extra=False):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                if extract_extra:
                    img = Image.open(abs_filename)
                    if not img:
                        continue
                    w, h = img.size
                    info = dict(filename=abs_filename, width=w, height=h, xmin=0, ymin=0, xmax=w, ymax=h)
                else:
                    info = dict(filename=abs_filename)
                inputs.append((int(base), info))
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


class ImagePatchIndex:
    def __init__(self, image_index, patch_index=0):
        self.image_index = image_index
        self.patch_index = patch_index


class IndexedPatchSampler(Sampler):
    """Samples patches across images sequentially by index, always in the same order.
    """

    def __init__(self, data_source):
        self.num_images = len(data_source)
        if data_source.patch_count:
            self.num_patches = data_source.patch_count
            self.patch_index = data_source.patch_index
        else:
            # fallback to indexing whole images from dataset
            print('Warning: Data source has no patch information, falling back to whole image indexing.')
            self.num_patches = 0
            self.patch_index = []

    def __iter__(self):
        if self.num_patches:
            for i in range(self.num_images):
                for j in self.patch_index[i]:
                    yield ImagePatchIndex(i, j)
        else:
            return iter(range(self.num_images))

    def __len__(self):
        return self.num_patches if self.num_patches else self.num_images


class RandomPatchSampler(Sampler):
    """Oversamples random patches from images in random order.
        Repeats the same image index multiple times in a row to sample 'repeat' times
        from the same image for big read efficiency gains.
    """
    def __init__(self, data_source, oversample=32, repeat=1):
        self.oversample = oversample//repeat * repeat
        self.repeat = repeat
        self.num_samples = len(data_source)

    def __iter__(self):
        # There are simpler/more compact ways of doing this, but why not have a somewhat
        # meaningful fake patch index?
        for to in range(self.oversample//self.repeat):
            samples = torch.randperm(self.num_samples).long()
            for image_index in samples:
                for ti in range(self.repeat):
                    fake_patch_index = to * self.repeat + ti
                    yield ImagePatchIndex(image_index, fake_patch_index)

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
            patch_size=[256, 256],
            patch_stride=128,
            transform=None,
            target_transform=None):

        extract_extra = False if os.path.exists(processing_file) else True
        input_ids, input_infos = find_inputs(
            input_root, types=['.jpg'], extract_extra=extract_extra)
        if len(input_ids) == 0:
            raise(RuntimeError("Found 0 images in : " + input_root))
        self.input_index = input_ids

        self.patch_index = [[]] * len(input_ids)
        self.patch_count = 0
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.data_by_id = defaultdict(dict)
        for index, (k, v) in enumerate(zip(input_ids, input_infos)):
            if 'width' in v:
                patch_info = self._calc_patch_info(v)
                num_patches = patch_info['num']
                self.patch_index[index] = list(range(num_patches))
                self.patch_count += num_patches
                v['patches'] = patch_info
            v['index'] = index
            self.data_by_id[k].update(v)

        self.has_targets = False
        if os.path.exists(target_root):
            targets = find_targets(target_root, input_ids, types=['.npz'])
            if len(targets):
                for k, v in targets.items():
                    self.data_by_id[k]['target'] = v
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
            cols = ['xmin', 'ymin', 'xmax', 'ymax', 'width', 'height']
            process_df[cols] = process_df[cols].astype(int)
            process_df['train_id'] = process_df.filename.map(lambda x: int(os.path.splitext(x)[0]))
            process_df.set_index(['train_id'], inplace=True)
            for k, v in process_df[cols].to_dict(orient='index').items():
                d = self.data_by_id[k]
                if d is not None:
                    patch_info = self._calc_patch_info(v)
                    num_patches = patch_info['num']
                    self.patch_index[d['index']] = list(range(num_patches))
                    self.patch_count += num_patches
                    v['patches'] = patch_info
                    d.update(v)
                    #print(d, self.patch_count)

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
                    xy_offset = np.array([d['xmin'], d['ymin']])
                    coords[:, :2] = coords[:, :2] + xy_offset
                    d['coords'] = coords

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
        self.ttime = utils.AverageMeter()

    def _calc_patch_info(self, input_info):
        x_min = input_info['xmin']
        x_max = input_info['xmax']
        y_min = input_info['ymin']
        y_max = input_info['ymax']
        assert y_max > y_min and x_max > x_min
        buffer_w = input_info['width']
        buffer_h = input_info['height']
        box_w = x_max - x_min
        box_h = y_max - y_min
        # FIXME switch to use bbox constraints
        num_patches, patch_cols, patch_rows = utils.calc_num_patches(
            buffer_w, buffer_h, self.patch_size, self.patch_stride)
        patch_origin_x = 0
        patch_origin_y = 0
        # if we have a bounding box border, see if we can squeeze an extra box in each dimension
        # if x_min != 0 or x_max != buffer_w:
        #     new_w = patch_cols * stride + patch_size[0]
        #     print(new_w, buffer_w)
        #     if new_w <= buffer_w:
        #         patch_cols += 1
        #         patch_origin_x = x_min - (new_w - box_w) // 2
        # if y_min != 0 or y_max != buffer_h:
        #     new_h = patch_rows * stride + patch_size[1]
        #     if new_h <= buffer_h:
        #         patch_rows += 1
        #         patch_origin_y = y_min - (new_h - box_h) // 2
        num_patches = patch_cols * patch_rows
        patch_info = dict(
            num=num_patches, cols=patch_cols, rows=patch_rows, origin_x=patch_origin_x, origin_y=patch_origin_y)
        return patch_info

    @functools.lru_cache(4)
    def _load_input(self, input_id):
        path = self.data_by_id[input_id]['filename']
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

    def _indexed_patch_center(self, input_id, patch_index):
        d = self.data_by_id[input_id]
        patch_info = d['patches']
        #patch_index = patch_index % patch_info['num']
        pc, pr = utils.index_to_rc(patch_index, patch_info['cols'])
        cx = pc * self.patch_stride + self.patch_size[0] // 2
        cy = pr * self.patch_stride + self.patch_size[1] // 2
        #print(patch_index, cx, cy)
        return cx, cy

    def _random_patch_center(self, input_id, w, h):
        d = self.data_by_id[input_id]
        if len(d['coords']) and random.random() < 0.4:
            # 40% of the time, randomly pick a point around an actual sealion
            cx, cy, _ = d['coords'][random.randint(0, len(d['coords']) - 1)]
            cx = cx + random.randint(-self.patch_size[0] // 4, self.patch_size[0] // 4)
            cy = cy + random.randint(-self.patch_size[1] // 4, self.patch_size[1] // 4)
        else:
            # return random center coords for specified patch size within a specified (x, y, w, h) bounding box
            pw, ph = self.patch_size[0] // 2, self.patch_size[1] // 2
            if 'xmin' in d:
                x_min = d['xmin']
                x_max = d['xmax']
                y_min = d['ymin']
                y_max = d['ymax']
                assert x_max <= w and x_max - x_min > 0
                assert y_max <= h and y_max - y_min > 0
            else:
                x_min = 0
                x_max = w
                y_min = 0
                y_max = h
            x_min += pw
            x_max -= pw
            y_min += ph
            y_max -= ph
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

        crop_w, crop_h = utils.calc_crop_size(self.patch_size[0], self.patch_size[1], angle, scale)
        input_tile = utils.crop_around(input_img, cx, cy, crop_w, crop_h)
        if transform_target:
            target_tile = utils.crop_around(target_arr, cx, cy, crop_w, crop_h)

        # Perform tile geometry transforms if needed
        if angle or scale != 1. or hflip or vflip:
            Mtrans = np.identity(3)
            Mtrans[0, 2] = (self.patch_size[0] - crop_w) // 2
            Mtrans[1, 2] = (self.patch_size[1] - crop_h) // 2
            if hflip:
                Mtrans[0, 0] *= -1
                Mtrans[0, 2] = self.patch_size[0] - Mtrans[0, 2]
            if vflip:
                Mtrans[1, 1] *= -1
                Mtrans[1, 2] = self.patch_size[1] - Mtrans[1, 2]

            if angle or scale != 1.:
                Mrot = cv2.getRotationMatrix2D((crop_w//2, crop_h//2), angle, scale)
                Mfinal = np.dot(Mtrans, np.vstack([Mrot, [0, 0, 1]]))
            else:
                Mfinal = Mtrans

            #print(input_tile.shape)
            input_tile = cv2.warpAffine(input_tile, Mfinal[:2, :], tuple(self.patch_size))
            if transform_target:
                tt64 = target_tile.astype(np.float64)
                tt64 = cv2.warpAffine(tt64, Mfinal[:2, :], tuple(self.patch_size))
                if scale != 1.:
                    tt64 /= scale**2
                target_tile = tt64.astype(np.float32)

        return input_tile, target_tile

    def __getitem__(self, index):
        if isinstance(index, ImagePatchIndex):
            patch_index = index.patch_index
            index = index.image_index
        else:
            patch_index = 0  #FIXME sort this out

        input_id = self.input_index[index % len(self)]
        input_img = self._load_input(input_id)
        #print(input_id, index, patch_index)
        h, w = input_img.shape[:2]
        if self.train:
            target_arr = self._load_target(input_id)
            test_patch = False
            if test_patch:
                print(w, h)
                num_patch = utils.calc_num_patches(w, h, self.patch_size, self.patch_stride)
                view, view_rc = utils.patch_view(target_arr, self.patch_size, self.patch_stride)
                assert num_patch[0] == view.shape[0] and num_patch[1] == view_rc[1] and num_patch[2] == view_rc[0]
                print('view', view.shape, view_rc, num_patch)
                reconstruct_target = np.zeros(target_arr.shape, dtype=target_arr.dtype)
                start = time.time()
                utils_cython.merge_patches_float32(
                    reconstruct_target, view, view_rc[1], self.patch_size, self.patch_stride)
                self.ttime.update(time.time() - start)
                print(self.ttime.val, self.ttime.avg)
                print(reconstruct_target.sum(), target_arr.sum())

            attempts = 16
            for i in range(attempts):
                tw, th = self.patch_size
                cx, cy = self._random_patch_center(input_id, w, h)
                input_patch, target_patch = self._crop_and_transform(cx, cy, input_img, target_arr, randomize=True)
                # check centre of chosen patch_index for valid pixels
                if np.any(utils.crop_around(input_patch, tw//2, th//2, tw//4, th//4)):
                    break
            input_tile_tensor = self.transform(input_patch)
            target_tile_tensor = to_tensor(target_patch)
        else:
            target_arr = None
            if self.has_targets:
                target_arr = self._load_target(input_id)

            test_patch = False
            if test_patch:
                def view_tensor(arr):
                    assert (isinstance(arr, np.ndarray))
                    t = torch.from_numpy(arr.transpose((0, 3, 1, 2)))
                    # print(t.size())
                    if isinstance(t, torch.ByteTensor):
                        return t.float().div(255)
                    return t

                view, view_rc = utils.patch_view(input_img, self.patch_size, self.patch_stride)
                print('view', view.shape, view_rc)
                viewt = view_tensor(view)
                tvutils.save_image(viewt, 'view-%d.jpg' % index, nrow=view_rc[1], normalize=True)

                reconstruct_img = np.zeros((h, w, 3), dtype=np.uint8)
                start = time.time()
                utils_cython.merge_patches_uint8(reconstruct_img, view, view_rc[1], self.patch_size, self.patch_stride)
                self.ttime.update(time.time() - start)
                print(self.ttime.val, self.ttime.avg)

                reconstruct_img = cv2.cvtColor(reconstruct_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite('borg-%d.jpg' % index, reconstruct_img)

            cx, cy = self._indexed_patch_center(input_id, patch_index)
            input_patch, target_patch = self._crop_and_transform(cx, cy, input_img, target_arr, randomize=False)
            input_tile_tensor = self.transform(input_patch)
            if target_patch is None:
                target_tile_tensor = torch.zeros(1)
            else:
                target_tile_tensor = to_tensor(target_patch)
            #print(input_tile_tensor.size(), target_tile_tensor)

        #cv2.imwrite('test-scaled-input-%d.png' % index, input_patch)
        #cv2.imwrite('test-scaled-target-%d.png' % index, 4096*target_tile[:, :, :3])

        index_tensor = torch.LongTensor([input_id, index, patch_index])

        return input_tile_tensor, target_tile_tensor, index_tensor

    def __len__(self):
        return len(self.input_index)

    def get_num_patches(self, input_id=None):
        if input_id is None:
            return self.patch_count
        else:
            if input_id in self.data_by_id:
                return self.data_by_id[input_id]['patches']['num']
            else:
                return 0

    def get_input_size(self, input_id):
        if input_id in self.data_by_id:
            d = self.data_by_id[input_id]
            return d['width'], d['height']
        else:
            return 0, 0

    def get_patch_cols(self, input_id):
        if input_id in self.data_by_id:
            return self.data_by_id[input_id]['patches']['cols']
        else:
            return 0