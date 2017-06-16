import argparse
import os
import time
import shutil
from dataset import SealionDataset, RandomTileSampler
from model_cnet import ModelCnet
from model_countception import ModelCountception

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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = parser.parse_args()

    train_input_root = os.path.join(args.data, 'Train-processed/inputs')
    train_target_root = os.path.join(args.data, 'Train-processed/targets')
    train_counts_file = os.path.join(args.data, 'Train/train.csv')

    batch_size = args.batch_size
    num_epochs = 1000
    tile_size = [256, 256]
    dataset = SealionDataset(train_input_root, train_target_root, train_counts_file, tile_size=tile_size)
    sampler = RandomTileSampler(dataset, oversample=256, repeat=8)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=True, num_workers=args.num_processes, sampler=sampler)
    model = ModelCnet(outplanes=5)
    if not args.no_cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_fn = torch.nn.L1Loss() #torch.nn.MSELoss()
    # optionally resume from a checkpoint

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(1, num_epochs + 1):
        adjust_learning_rate(optimizer, epoch, initial_lr=args.lr, decay_epochs=3)
        train_epoch(epoch, model, loader, optimizer, loss_fn, no_cuda=args.no_cuda)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'c-net',
            'state_dict':  model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=True)


def train_epoch(epoch, model, loader, optimizer, loss_fn, log_interval=10, no_cuda=False):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        data_time_m.update(time.time() - end)
        if no_cuda:
            input_var, target_var = autograd.Variable(input), autograd.Variable(target)
        else:
            input_var, target_var = autograd.Variable(input.cuda()), autograd.Variable(target.cuda())

        output = model(input_var)
        loss = loss_fn(output, target_var)
        losses_m.update(loss.data[0], input_var.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time_m.update(time.time() - end)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  '
                  'Loss: {loss.val:.6f} ({loss.avg:.4f})  '
                  'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                  '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch,
                batch_idx * len(input), len(loader.sampler),
                100. * batch_idx / len(loader),
                loss=losses_m,
                batch_time=batch_time_m,
                rate=input_var.size(0) / batch_time_m.val,
                rate_avg=input_var.size(0) / batch_time_m.avg,
                data_time=data_time_m))
        end = time.time()


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs=5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
