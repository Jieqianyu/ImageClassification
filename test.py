import argparse
import os
import time
from utils import *
import cfg
from datasets import ImageFolder
from model import ConvNet

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top3],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(args.device)
            target = target.to(args.device)

            # compute output
            output  = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc3 = accuracy(output, target, topk=(1,3))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top3.update(acc3.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

    return losses.avg, top1.avg


def main(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(args.device))

    # create model
    model = ConvNet(cfg.NUM_CLASSES).to(args.device)
    
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # load checkpoint
    if args.model_weight:
        if os.path.isfile(args.model_weight):
            print("=> loading checkpoint '{}'".format(args.model_weight))
            
            checkpoint = torch.load(args.model_weight, map_location=args.device)
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}'".format(args.model_weight))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_weight))

    # Data loading code
    test_dataset = ImageFolder(cfg.TEST_PATH, mode='test')

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # Evaluate on test dataset
    validate(test_loader, model, criterion, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 4), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-w', '--model-weight', default=cfg.WEIGHTS_PATH, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: {})'.format(cfg.WEIGHTS_PATH))

    args = parser.parse_args()
    main(args)
