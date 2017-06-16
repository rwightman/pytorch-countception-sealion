import argparse
import os
from dataset import SealionDataset, RandomTileSampler
from model import Model

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

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
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')


def train_epoch(epoch, model, loader, optimizer, loss_fn, log_interval=10, no_cuda=False):
    model.train()
    pid = os.getpid()
    for batch_idx, (input, target) in enumerate(loader):
        if no_cuda:
            input, target = autograd.Variable(input), autograd.Variable(target)
        else:
            input, target = autograd.Variable(input.cuda()), autograd.Variable(target.cuda())
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(input), len(loader.sampler),
                100. * batch_idx / len(loader), loss.data[0]))


def main():
    args = parser.parse_args()

    train_input_root = os.path.join(args.data, 'Train-processed/inputs')
    train_target_root = os.path.join(args.data, 'Train-processed/targets')
    train_counts_file = os.path.join(args.data, 'Train/train.csv')

    batch_size = args.batch_size
    num_epochs = 1000
    tile_size = [284, 284]
    dataset = SealionDataset(train_input_root, train_target_root, train_counts_file, tile_size=tile_size)
    sampler = RandomTileSampler(dataset, oversample=256, repeat=8)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=True, num_workers=args.num_processes, sampler=sampler)
    model = Model()
    if not args.no_cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        train_epoch(epoch, model, loader, optimizer, loss_fn, no_cuda=args.no_cuda)


if __name__ == '__main__':
    main()
