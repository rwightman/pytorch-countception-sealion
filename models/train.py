import argparse
import os
from dataset import SealionDataset

import torch

parser = argparse.ArgumentParser(description='PyTorch Sealion count training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 2)')


def main():
    args = parser.parse_args()

    train_input_root = os.path.join(args.data, 'Train-processed/inputs')
    train_target_root = os.path.join(args.data, 'Train-processed/targets')
    train_counts_file = os.path.join(args.data, 'Train/train.csv')

    dataset = SealionDataset(train_input_root, train_target_root, train_counts_file)

    for x in range(20):
        input, target = dataset[x]




if __name__ == '__main__':
    main()