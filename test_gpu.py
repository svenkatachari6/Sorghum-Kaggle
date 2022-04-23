import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import argparse
import os
import shutil
import numpy as np
import time
import utils
import models

# from logger import Logger


parser = argparse.ArgumentParser(description='CURE-TSR Evaluation')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


def main():
    global args 
    args = parser.parse_args()

    debug = 1  # 0: normal mode 1: debug mode

    # Data loading code
    # args.data: path to the dataset

    testdir = os.path.join(args.data, 'ChallengeFree')

    test_dataset = utils.CURETSRDataset(testdir, transforms.Compose([
        transforms.Resize([28, 28]), transforms.ToTensor(), utils.l2normalize, utils.standardization]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    
    USE_CUDA = True
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = models.Net()
    # model = models.SoftmaxClassifier()
    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)
    print("=> creating model %s " % model.__class__.__name__)

    # savedir = 'CNN_iter/checkpoint.pth.tar'
    savedir = 'model_best.pth.tar'
    checkpointdir = os.path.join('checkpoints', savedir)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    print(os.path.isfile('./checkpoints'), checkpointdir)

    # if os.path.isfile(checkpointdir):
    print("=> loading checkpoint '{}'".format(checkpointdir))
    checkpoint = torch.load(checkpointdir)
    # checkpoint = torch.load(checkpointdir, map_location=torch.device('cpu'))
    # checkpoint = load_checkpoint(checkpointdir)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {}, best_prec1 @ Source {})"
            .format(checkpointdir, checkpoint['epoch'], best_prec1))
    # else:
    #     print("=> no checkpoint found at '{}'".format(checkpointdir))

    


    
    evaluate(test_loader, model, criterion)
    return


def evaluate(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    acc = AverageMeter()
    USE_CUDA = True
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var).cuda()
        loss = criterion(output, target_var.cuda())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        batch_acc = batch_accuracy(output.data, target)
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        acc.update(batch_acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Test: [{i}/{len(test_loader)}]\t'
                  f'Time {batch_time.val} ({batch_time.avg})\t'
                  f'Loss {losses.val} ({losses.avg})\t'
                  f'Prec@1 {top1.val} ({top1.avg})\t'
                  f'Prec@5 {top5.val} ({top5.avg})\t'
                  f'Prec@5 {acc.val} ({acc.avg})')

    print(f' * Prec@1 {top1.avg} Prec@5 {top5.avg}')

    return losses.avg, top1.avg, top5.avg


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []

    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def batch_accuracy(output, target):
    """Computes the accuracy for a batch"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum()

        return correct * 100.0 / batch_size

# Reference: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/torchtools.py#L104

def load_checkpoint(fpath):
    r"""Loads checkpoint.
    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.
    Args:
        fpath (str): path to checkpoint.
    Returns:
        dict
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not os.path.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


if __name__ == '__main__':
    main()
