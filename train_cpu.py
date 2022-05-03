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


parser = argparse.ArgumentParser(description='Sorghum Dataset Training and Evaluation')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--clip-grad', '--clip-grad', default=0.1, type=float,
                    metavar='C', help='clip gradient (default: 0.5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main():
    global args 
    args = parser.parse_args()

    debug = 0 # 0: normal mode 1: debug mode

    # Data loading code
    # args.data: path to the dataset

    traindir = os.path.join(args.data, 'train_images')

    # Load training and validation from the same folder using Sampler
    alex_net = False
    sm20 = False
    if alex_net:
        dataset = utils.SorghumDataset(traindir, transforms.Compose([
            transforms.Resize([256, 256]), 
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(), 
            utils.l2normalize, 
            utils.standardization
            ]))
    elif sm20:
        dataset = utils.SorghumDataset(traindir, transforms.Compose([
            transforms.Resize([256, 256]), 
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(), 
            utils.l2normalize, 
            utils.standardization
            ]))
    else:
        dataset = utils.SorghumDataset(traindir, transforms.Compose([
            transforms.Resize([256, 256]), 
            transforms.ToTensor(), 
            utils.l2normalize, 
            utils.standardization
            ]))

    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler = train_sampler, 
                                                num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler = valid_sampler, 
                                                num_workers=args.workers, pin_memory=True)

    
    USE_CUDA = False
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = models.Net()
    # model = models.SoftmaxClassifier()
    # model = models.AlexNetSoftMax()
    # model = models.Softmax20()

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)
    print("=> creating model %s " % model.__class__.__name__)

    savedir = 'CNN_iter'
    checkpointdir = os.path.join('checkpoints', savedir)

    if not debug:
        if not os.path.exists(checkpointdir):
            os.makedirs(checkpointdir)
        print('log directory: %s' % os.path.join('./logs', savedir))
        print('checkpoints directory: %s' % checkpointdir)

    # Set the logger
    # if not debug:
    #     logger = Logger(os.path.join('./logs/', savedir))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters())

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}, best_prec1 @ Source {})"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        evaluate(test_loader, model, criterion)
        return

    # cudnn.benchmark = True

    timestart = time.time()
    best_prec1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        print('\n*** Start Training *** \n')
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        print('\n*** Start Testing *** \n')
        test_loss, test_prec1, _ = evaluate(test_loader, model, criterion)

        info = {
            'Testing loss': test_loss,
            'Testing Accuracy': test_prec1
        }

        # remember best prec@1 and save checkpoint
        is_best = test_prec1 > best_prec1
        best_prec1 = max(test_prec1, best_prec1)

        if is_best:
            best_epoch = epoch + 1

        if not debug:
            for tag, value in info.items():
                # logger.scalar_summary(tag, value, epoch+1)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'last_prec1': test_prec1,
                    'optimizer': optimizer.state_dict()}, is_best, checkpointdir)


    print('Best epoch: ', best_epoch)
    print('Total processing time: %.4f' % (time.time() - timestart))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    acc = AverageMeter()

    USE_CUDA = False
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        # print(output)
        loss = criterion(output, target_var)
        # print (model.fc.weight.data)
        # if np.isnan(loss.item()):
        # # if True:
        #     print(output, target_var)
        #     print(loss)
        # if torch.sum(torch.isnan(output)):
        #     print(torch.isnan(input_var))
        #     print('here')
    
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
        # measure accuracy and record loss
        
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        batch_acc = batch_accuracy(output.data, target)
        losses.update(loss.data, input.size(0))  # input.size(0): Batch size
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        acc.update(batch_acc, input.size(0))

        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss.backward()
        # print(args.clip_grad)
        # nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val} ({batch_time.avg})\t'
                  f'Data {data_time.val} ({data_time.avg})\t'
                  f'Loss {losses.val} ({losses.avg})\t'
                  f'Prec@1 {top1.val} ({top1.avg})\t'
                  f'Prec@5 {top5.val} ({top5.avg})\t'
                  f'Prec@5 {acc.val} ({acc.avg})')
    
    return losses.avg, top1.avg, top5.avg


def evaluate(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    acc = AverageMeter()
    USE_CUDA = False
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target.to(device)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

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


def save_checkpoint(state, is_best, checkpointdir):
    fullpath = os.path.join(checkpointdir, 'checkpoint.pth.tar')
    fullpath_best = os.path.join(checkpointdir, 'model_best.pth.tar')
    torch.save(state, fullpath)

    if is_best:
        shutil.copyfile(fullpath, fullpath_best)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
