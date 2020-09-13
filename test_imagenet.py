import argparse
import logging
import os
import pprint
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from configs.imagenet_val_cfg import cfg
from dataset.imagenet_scripts import imagenet_data
from tools import utils
from tools.multadds_count import comp_multadds
import genotypes
from run_apis.trainer import Trainer
from model import NetworkImageNet as Network

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Params")
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--arch', type=str, default='RelativeNAS', help='which architecture to use')
    parser.add_argument('--data_path', type=str, default='/raid/huangsh/imagenet/ILSVRC2012/lmdb/',
                        help='location of the dataset')
    parser.add_argument('--gpus', type=str, default='14,15', help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=46, help='num of init channels')
    parser.add_argument('--layers', type=int, default=14, help='total number of layers')
    parser.add_argument('--model_path', type=str,
                        default="/raid/huangsh/code/CSO/exp_dirs/eval-imagenet-archCSO_CIFAR10-l14-c46-lr0.5/weights_249_75.1179999194336.pt",
                        help='path of pretrained model')
    parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
    parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    config = cfg

    CLASSES = 1000
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpus)

    args.save = os.path.join(args.save, 'output')
    utils.create_exp_dir(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled = True

    logging.info("args = %s", args)
    logging.info('Training with config:')
    logging.info(pprint.pformat(config))

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model.drop_path_prob = args.drop_path_prob
    model = nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path))

    imagenet = imagenet_data.ImageNet12(trainFolder=os.path.join(args.data_path, 'train'),
                                        testFolder=os.path.join(args.data_path, 'val'),
                                        num_workers=config.data.num_workers, data_config=config.data)
    valid_queue = imagenet.getTestLoader(config.data.batch_size)
    trainer = Trainer(None, valid_queue, None, config, args.report_freq)

    with torch.no_grad():
        val_acc_top1, val_acc_top5, valid_obj, batch_time = trainer.infer(model)
