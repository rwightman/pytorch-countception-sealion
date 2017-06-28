import argparse
import os
import time
import shutil
from datetime import datetime
from dataset import SealionDataset, RandomPatchSampler
from models import ModelCnet, ModelCountception
from utils import AverageMeter, get_outdir

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torch.optim as optim
import torchvision.utils

parser = argparse.ArgumentParser(description='PyTorch Sealion count training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='countception', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--loss', default='l1', type=str, metavar='LOSS',
                    help='Loss function (default: "l1"')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='M',
                    help='weight decay (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-batches', action='store_true', default=False,
                    help='save images of batch inputs and targets every log interval for debugging/verification')


def main():
    args = parser.parse_args()

    train_input_root = os.path.join(args.data, 'inputs')
    train_target_root = os.path.join(args.data, 'targets')
    train_process_file = os.path.join(args.data, 'processed.csv')
    train_counts_file = './data/correct_train.csv'
    train_coords_file = './data/correct_coordinates.csv'
    output_dir = get_outdir('./output', 'train', datetime.now().strftime("%Y%m%d-%H%M%S"))

    batch_size = args.batch_size
    num_epochs = 1000
    patch_size = [256] * 2
    num_outputs = 5
    target_type = 'countception' if args.model == 'countception' or args.model == 'cc' else 'density'
    debug_model = False

    torch.manual_seed(args.seed)

    dataset = SealionDataset(
        train_input_root,
        train_target_root,
        train_counts_file,
        train_coords_file,
        train_process_file,
        train=True,
        patch_size=patch_size,
        target_type=target_type,
        generate_target=True,
        per_image_norm=True
    )

    sampler = RandomPatchSampler(dataset, oversample=32, repeat=16)

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=True, num_workers=args.num_processes, sampler=sampler)

    if args.model == 'cnet':
        model = ModelCnet(outplanes=num_outputs, target_size=patch_size, debug=debug_model)
    elif args.model == 'countception' or args.model == 'cc':
        model = ModelCountception(outplanes=num_outputs, debug=debug_model)
    else:
        assert False and "Invalid model"

    if not args.no_cuda:
        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        else:
            model.cuda()

    if args.opt.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        assert False and "Invalid optimizer"

    if args.loss.lower() == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif args.loss.lower() == 'smoothl1':
        loss_fn = torch.nn.SmoothL1Loss()
    elif args.loss.lower() == 'mse':
        loss_fn = torch.nn.MSELoss()
    else:
        assert False and "Invalid loss function"

    # optionally resume from a checkpoint
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(start_epoch, num_epochs + 1):
        adjust_learning_rate(optimizer, epoch, initial_lr=args.lr, decay_epochs=3)
        train_epoch(epoch, model, loader, optimizer, loss_fn, args, output_dir)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model.name(),
            'state_dict':  model.state_dict(),
            'optimizer': optimizer.state_dict(),
            },
            is_best=False,
            filename='checkpoint-%d.pth.tar' % epoch,
            output_dir=output_dir)


def train_epoch(epoch, model, loader, optimizer, loss_fn, args, output_dir):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (input, target, index) in enumerate(loader):
        data_time_m.update(time.time() - end)
        if args.no_cuda:
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
        if batch_idx % args.log_interval == 0:
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

            if args.save_batches:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'input-batch-%d.jpg' % batch_idx),
                    normalize=True)
                torchvision.utils.save_image(
                    torch.sum(target, dim=1),
                    os.path.join(output_dir, 'target-batch-%d.jpg' % batch_idx),
                    normalize=True)
        end = time.time()


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs=5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', output_dir=''):
    save_path = os.path.join(output_dir, filename)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(output_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
