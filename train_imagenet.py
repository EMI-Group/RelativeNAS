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

from dataset.imagenet_scripts import imagenet_data
from torch.autograd import Variable
import genotypes

from tools.lr_scheduler import get_lr_scheduler
from run_apis.trainer import Trainer
from tools.utils import cross_entropy_with_label_smoothing
from model import NetworkImageNet as Network
from configs.imagenet_train_cfg import cfg as config
from tools.utils import count_parameters_in_MB
from tools.multadds_count import comp_multadds

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')  # '11,12,13,14'
parser.add_argument('--arch', type=str, default='PairNAS', help='which architecture to use')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--data_path', type=str, default='/raid/huangsh/imagenet/ILSVRC2012/lmdb')
# model setting
parser.add_argument('--init_channels', type=int, default=46, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpus)
args.save = '../exp_dirs/eval-imagenet-arch{}-l{}-c{}-lr{}'.format(args.arch, args.layers, args.init_channels,
                                                                   config.optim.init_lr)
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
NUM_CLASSES = 1000


def adjust_lr(optimizer, epochs, learning_rate, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if epochs - epoch > 5:
        lr = learning_rate * (epochs - 5 - epoch) / (epochs - 5)
    else:
        lr = learning_rate * (epochs - epoch) / ((epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, NUM_CLASSES, args.layers, config.optim.auxiliary, genotype)

    start_epoch = 0
    model.eval()
    model.drop_path_prob = args.drop_path_prob * 0
    # compute the params as well as the multi-adds
    params = count_parameters_in_MB(model)
    logging.info("Params = %.2fMB" % params)
    mult_adds = comp_multadds(model, input_size=config.data.input_size)
    logging.info("Mult-Adds = %.2fMB" % mult_adds)

    model.train()
    if len(args.gpus) > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    if config.optim.label_smooth:
        criterion = CrossEntropyLabelSmooth(NUM_CLASSES, config.optim.smooth_alpha)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), config.optim.init_lr, momentum=config.optim.momentum,
                                weight_decay=config.optim.weight_decay)

    imagenet = imagenet_data.ImageNet12(trainFolder=os.path.join(args.data_path, 'train'),
                                        testFolder=os.path.join(args.data_path, 'val'),
                                        num_workers=config.data.num_workers,
                                        type_of_data_augmentation=config.data.type_of_data_aug, data_config=config.data,
                                        size_images=config.data.input_size[1], scaled_size=config.data.scaled_size[1])
    train_queue, valid_queue = imagenet.getTrainTestLoader(config.data.batch_size)

    if config.optim.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.train_params.epochs))

    trainer = Trainer(train_queue, valid_queue, criterion, config, args.report_freq)
    best_epoch = [0, 0, 0]  # [epoch, acc_top1, acc_top5]
    lr = config.optim.init_lr
    for epoch in range(start_epoch, config.train_params.epochs):
        if config.optim.lr_schedule == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif config.optim.lr_schedule == 'linear':  # with warmup initial
            optimizer, current_lr = adjust_lr(optimizer, config.train_params.epochs, lr, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)
        if epoch < 5:  # Warmup epochs for 5
            current_lr = lr * (epoch + 1) / 5.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)

        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if len(args.gpus) > 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / config.train_params.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / config.train_params.epochs
        train_acc_top1, train_acc_top5, train_obj, batch_time, data_time = trainer.train(model, optimizer, epoch)
        with torch.no_grad():
            val_acc_top1, val_acc_top5, batch_time, data_time = trainer.infer(model, epoch)
        if val_acc_top1 > best_epoch[1]:
            best_epoch = [epoch, val_acc_top1, val_acc_top5]
            if epoch >= 0:  # 120
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_acc_top1': val_acc_top1,
                    'optimizer': optimizer.state_dict(),
                }, save_path=args.save, epoch=epoch, is_best=True)
                if len(args.gpus) > 1:
                    utils.save(model.module.state_dict(),
                               os.path.join(args.save, 'weights_{}_{}.pt'.format(epoch, val_acc_top1)))
                else:
                    utils.save(model.state_dict(),
                               os.path.join(args.save, 'weights_{}_{}.pt'.format(epoch, val_acc_top1)))

        logging.info('BEST EPOCH %d  val_top1 %.2f val_top5 %.2f', best_epoch[0], best_epoch[1], best_epoch[2])
        logging.info('epoch: {} \t train_acc_top1: {:.4f} \t train_loss: {:.4f} \t val_acc_top1: {:.4f}'.format(epoch,
                                                                                                                train_acc_top1,
                                                                                                                train_obj,
                                                                                                                val_acc_top1))

    logging.info("Params = %.2fMB" % params)
    logging.info("Mult-Adds = %.2fMB" % mult_adds)


if __name__ == '__main__':
    main()
