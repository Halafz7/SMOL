import os, sys, shutil, time, random
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import numpy as np
import random
from models import *

parser = argparse.ArgumentParser(description='General PyTorch training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lmbda', type=float, default=1e-07)
parser.add_argument('--t', type=int, default=350)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', type=str, default='../../..', help='Path to dataset')
parser.add_argument('--p_init', type=float, default=8, help='value to initialize p with')
parser.add_argument('--baseline', dest='baseline', action='store_true', help='trains a baseline FP model')

parser.add_argument('--model', type=str, default='preresnet', help='model architecture (resnet or preresnet)')
parser.add_argument('--fp_layers', type=str, default='shortcuts' ,help='which layers to set as full precision (all, shortcuts, or ends')
parser.add_argument('--shortcut_type', type=str, default='CB', help='which shortcut type to use (P, C, or CB')
parser.add_argument('--val', dest='val', action='store_true', help='create and use validation set from training set')

parser.add_argument('--prune', type=int, default=0, help='value to initialize p with')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=650)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test-batch_size', type=int, default=1000)

parser.add_argument('--sgdlr', type=float, default=0.1)
parser.add_argument('--adamlr', type=float, default=1e-3)

parser.add_argument('--steps', type=int, default=80)
parser.add_argument('--lrdecay', type=float, default=10)

parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--qf', type=str, default='floor')

# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./snapshots/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--nosave', dest='nosave', action='store_true', help='do not save models')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='only evaluate the model')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')

# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--job-id', type=str, default='default')
args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
job_id = args.job_id
args.save_path = args.save_path + job_id
result_png_path = './results/' + job_id + '.png'
if not os.path.isdir('results'): os.mkdir('results')  
    
out_str = str(args)
print(out_str)

if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

best_acc = 0

def load_dataset():
    mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)

    if args.val:
        train_data, test_data = torch.utils.data.random_split(train_data, [45000, 5000])
    else:
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = 10
    return num_classes, train_loader, test_loader

def load_model(num_classes, log):
    if args.model == 'preresnet':  model = PreResNet(p_init=args.p_init, fp_layers=args.fp_layers, shortcut_type=args.shortcut_type, prune=args.prune)

    print_log("=> network :\n {}".format(model), log)
    model = torch.nn.DataParallel(model.cuda(), device_ids=list(range(args.ngpu)))
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in trainable_params])
    print_log("Number of parameters: {}".format(params), log)
    return model

def main():
    global best_acc

    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_{}_seed_{}.txt'.format(args.job_id,args.manualSeed)), 'w')
    aux_log = open(os.path.join(args.save_path, 'log_aux_{}_seed_{}.txt'.format(args.job_id,args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    if not os.path.isdir(args.data_path): os.makedirs(args.data_path)

    num_classes, train_loader, test_loader = load_dataset()
    model = load_model(num_classes, log)
    
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    weight_params = []
    prec_params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.affine:
                weight_params.append(m.weight)
                weight_params.append(m.bias)
        elif isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear):
            weight_params.append(m.weight)
            prec_params.append(m.weight_s)
            if m.bias is not None:
                weight_params.append(m.bias)
                prec_params.append(m.bias_s)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            weight_params.append(m.weight)
            if m.bias is not None:
                weight_params.append(m.bias)

    print("Optimizing {} weight params and {} precision params".format(len(weight_params), len(prec_params)))

    if args.qf == 'floor':
        model.module.set_qf(torch.floor)
    elif args.qf == 'round':
        model.module.set_qf(torch.round)
    else:
        model.module.set_qf(torch.ceil)

    optimizers = []
    if len(prec_params)>0:
        optimizers.append(torch.optim.Adam(prec_params, lr=args.adamlr))
    if len(weight_params)>0:
        optimizers.append(torch.optim.SGD(weight_params, lr=args.sgdlr, momentum=args.momentum, weight_decay=args.weight_decay))

    recorder = RecorderMeter(args.epochs)

    if args.resume:
        if args.resume == 'auto':
            args.resume = os.path.join(args.save_path, 'model_best.pth.tar')
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            recorder.refresh(args.epochs)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizers[0].load_state_dict(checkpoint['optimizer1'])
            optimizers[-1].load_state_dict(checkpoint['optimizer2'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint", log)

    if args.evaluate:
        model.module.set_mode('quant')
        prec_string = model.module.print_precs()
        print_log(prec_string, log)
        validate(test_loader, model, criterion, log)
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    for epoch in range(args.start_epoch, args.epochs):
        current_lr = adjust_lr(optimizers, epoch)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [lr={:6.8f}]'.format(time_string(), epoch, args.epochs, need_time, current_lr) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        train_acc, train_los = train(train_loader, model, criterion, optimizers, epoch, log, aux_log)

        prec_string = model.module.print_precs()
        print_log(prec_string, log)
        
        val_acc, val_los = validate(test_loader, model, criterion, log)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc

        f_name = 'checkpoint_{}.pth.tar'.format(epoch)
        save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'recorder': recorder,
          'optimizer1' : optimizers[0].state_dict(),
          'optimizer2' : optimizers[-1].state_dict()

        }, is_best, args.save_path, f_name)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(result_png_path)

    log.close()

def train(train_loader, model, criterion, optimizers, epoch, log, aux_log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        if epoch < args.t:
            model.module.set_mode('noisy')
        else:
            model.module.set_mode('quant')

        output = model(input)
        loss = criterion(output, target)

        if epoch < args.t:
            loss += args.lmbda * model.module.prec_cost()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        for opt in optimizers: opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            if not (isinstance(opt, torch.optim.Adam) and epoch > args.t):
                opt.step()

        if not args.baseline:
            model.module.project()
        
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    print_log('  **Test**  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss {losses.avg:.5f} '.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses), log)
    return top1.avg, losses.avg

def print_log(print_string, log, verbose=True):
    if verbose: print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    if args.nosave: return
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_lr(optimizers, epoch):
    if epoch < 250:
        decay_factor = 1.
    elif epoch < 500:
        decay_factor = 10.
    elif epoch < 600:
        decay_factor = 100.
    else:
        decay_factor = 1000.

    sgdlr = args.sgdlr / decay_factor
    adamlr = args.adamlr / decay_factor
    for opt in optimizers[:2]:
        if isinstance(opt, torch.optim.Adam): lr = adamlr
        else: lr = sgdlr
        for param_group in opt.param_groups: param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__': main()
