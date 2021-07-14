import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from slow_fast_learning import init_pop, cal_center, gen_pairs, decode, update_state_dict, genotype

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--set', type=str, default="cifar10", help='data set')
parser.add_argument('--pop_size', type=int, default=20, help='population size')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=7, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
if args.set == 'cifar100':
    CIFAR_CLASSES = 100


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    logging.info("CIFAR_CLASSES = %s", CIFAR_CLASSES)

    assert args.pop_size % 2 == 0

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # initial population matrix
    p, lu = init_pop(args)

    # print the initial architectures in the population
    for j in range(args.pop_size):
        x = p[j]
        logging.info(str(genotype(x)))

    # initial a weight set using state_dict
    arch = p[0]
    aux_model = decode(args, CIFAR_CLASSES, arch, 0)
    state_dict = aux_model.state_dict()

    # matrix v is used to store the second derivatives
    v = np.zeros_like(p)

    optimizer = torch.optim.SGD(
        aux_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.set == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=False, transform=train_transform)

    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)

    # divide the training set into training set and validation set
    num_train = len(train_data)
    indices = list(range(num_train))
    import random
    random.shuffle(indices)
    random.shuffle(indices)
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # generate random pairs
        rpairs = gen_pairs(args)

        # calculate the center position
        center, var = cal_center(args, p)
        logging.info('the diversity is %e', var)

        # do paired learning
        # mask is used to indicate the index of teachers
        mask = np.zeros(args.pop_size//2, dtype=np.int_)
        for i in range(args.pop_size//2):
            arch_a = p[rpairs[i, 0], :]
            arch_b = p[rpairs[i, 1], :]

            # decode the architecture vectors into networks
            model_a = decode(args, CIFAR_CLASSES, arch_a, epoch)
            model_b = decode(args, CIFAR_CLASSES, arch_b, epoch)

            # training
            model_a, train_acc_a, _ = train(train_queue, model_a, state_dict, criterion, lr)
            model_b, train_acc_b, _ = train(train_queue, model_b, state_dict, criterion, lr)

            # validation
            valid_acc_a, valid_loss_a = infer(valid_queue, model_a, criterion)
            valid_acc_b, valid_loss_b = infer(valid_queue, model_b, criterion)

            mask[i] = (valid_loss_a > valid_loss_b)

            if valid_loss_a < valid_loss_b:
                state_dict = update_state_dict(state_dict, model_a, model_b)
            else:
                state_dict = update_state_dict(state_dict, model_b, model_a)

            logging.info('model_a: %d model_b: %d mask:%d', rpairs[i, 0], rpairs[i, 1], mask[i])
            logging.info('valid_acc comp %f %f', valid_acc_a, valid_acc_b)
            logging.info('valid_loss comp %f %f', valid_loss_a, valid_loss_b)

        # get the matrix of students and teachers
        students = mask * rpairs[:, 0] + np.logical_not(mask) * rpairs[:, 1]
        teachers = np.logical_not(mask) * rpairs[:, 0] + mask * rpairs[:, 1]

        # random matrix
        randco1 = np.random.rand(p.shape[0] // 2, p.shape[1])
        randco2 = np.random.rand(p.shape[0] // 2, p.shape[1])

        # students learn from teachers
        v[students, :] = randco1 * v[students, :] + randco2 * (p[teachers, :] - p[students, :])
        p[students, :] = p[students, :] + v[students, :]

        # boundary control
        for i in range(args.pop_size // 2):
            p[students[i], :] = np.maximum(p[students[i], :], lu[0, :])
            p[students[i], :] = np.minimum(p[students[i], :], lu[1, :])

        # print the architectures of the population
        for j in range(args.pop_size):
            x = p[j]
            logging.info(str(genotype(x)))

        scheduler.step()


def train(train_queue, model, state_dict, criterion, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    # model load weights from the weight set
    model.load_state_dict(state_dict, strict=True)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)

        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        # if step % args.report_freq == 0:
        #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return model, top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        # if step % args.report_freq == 0:
        #     logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
