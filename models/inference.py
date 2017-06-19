import argparse
import os
import time
import shutil
import numpy as np
import pandas as pd
from dataset import SealionDataset, SequentialTileSampler
from model_cnet import ModelCnet
from model_countception import ModelCountception
from utils import AverageMeter

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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

COLS = ['test_id', 'adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']


def main():
    args = parser.parse_args()

    test_input_root = os.path.join(args.data, 'Test')

    batch_size = args.batch_size
    tile_size = [256, 256]
    dataset = SealionDataset(
        test_input_root,
        train=False,
        tile_size=tile_size)
    sampler = SequentialTileSampler(dataset)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=args.num_processes, sampler=sampler)
    model = ModelCnet(outplanes=5)
    model.cuda()
    model.eval()

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    results = []
    try:
        for batch_idx, (input, target) in enumerate(loader):
            data_time_m.update(time.time() - end)
            #input_var, target_var = autograd.Variable(input.cuda()), autograd.Variable(target.cuda())
            #output = model(input_var)
            print(batch_idx)
            for x in target:
                counts = [0] * 5
                results.append([x[0]] + counts)
    except KeyboardInterrupt:
        pass
    results_df = pd.DataFrame(results, columns=COLS)
    results_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
