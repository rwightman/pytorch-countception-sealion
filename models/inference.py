import argparse
import os
import time
import shutil
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from dataset import SealionDataset, IndexedPatchSampler
from model_cnet import ModelCnet
from model_countception import ModelCountception
from utils import AverageMeter
from utils_cython  import merge_patches_float32
import torch
import torch.autograd as autograd
import torch.utils.data as data
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch Sealion count inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('-r', '--restore_checkpoint', default=None,
                    help='path to restore checkpoint, e.g. ./checkpoint-1.tar')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

COLS = ['test_id', 'adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']


def main():
    args = parser.parse_args()

    test_input_root = os.path.join(args.data, 'Test')

    batch_size = args.batch_size
    patch_size = [384] * 2
    dataset = SealionDataset(
        test_input_root,
        train=False,
        patch_size=patch_size)
    sampler = IndexedPatchSampler(dataset)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=args.num_processes, sampler=sampler)
    model = ModelCnet(outplanes=5)
    if not args.no_cuda:
        model.cuda()

    if args.restore_checkpoint is not None:
        assert os.path.isfile(args.restore_checkpoint), '%s not found' % args.restore_checkpoint
        checkpoint = torch.load(args.restore_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        #step = model.load(args.restore_checkpoint)
        print('Model restored from file: %s' % args.restore_checkpoint)

    model.eval()

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    current_id = -1
    patches = []
    results = []
    last_batch = -1
    last_index = None
    try:
        for batch_idx, (input, target, index) in enumerate(loader):
            data_time_m.update(time.time() - end)
            if not args.no_cuda:
                input_var, target_var = autograd.Variable(input.cuda()), autograd.Variable(target.cuda())
            else:
                input_var, target_var = autograd.Variable(input), autograd.Variable(target)
            output = model(input_var)
            output = output.permute(0, 2, 3, 1)
            output = output.cpu().data.numpy()

            for result_index, o in zip(index, output):
                input_id, index, patch_index = result_index
                #print('input_id, index, patch_index: ', input_id, index, patch_index)
                if current_id == -1:
                    current_id = input_id
                elif current_id != input_id:
                    # reconstruct output image from patches
                    w, h = dataset.get_input_size(input_id)
                    cols = dataset.get_patch_cols(input_id)
                    output_arr = np.zeros((h, w, 5), dtype=np.float32)
                    patches_arr = np.stack(patches)
                    merge_patches_float32(output_arr, patches_arr, cols, dataset.patch_size, dataset.patch_stride)
                    counts = list(np.sum(output_arr, axis=(0, 1)) / 1024.)
                    print(counts)
                    debug_image = True
                    if debug_image:
                        debug_img = cv2.normalize(
                            output_arr[:, :, :3], None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                        cv2.imwrite('output-%d.png' % current_id, debug_img)
                    results.append([current_id] + counts)
                    patches = []
                    current_id = input_id
                patches.append(o)
    except KeyboardInterrupt:
        pass
    results_df = pd.DataFrame(results, columns=COLS)
    results_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
