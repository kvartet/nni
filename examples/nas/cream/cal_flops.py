import os
import sys
import datetime
import torch
import numpy as np
import torch.nn as nn

# import timm packages
from timm.loss import LabelSmoothingCrossEntropy
from timm.data import Dataset, create_loader
from timm.models import resume_checkpoint

# import apex as distributed package
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    USE_APEX = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    USE_APEX = False

# import models and training functions
from lib.utils.flops_table import FlopsEst
from lib.models.structures.supernet import gen_supernet
from lib.config import DEFAULT_CROP_PCT, IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from lib.utils.util import parse_config_args, get_logger, \
    create_optimizer_supernet, create_supernet_scheduler

from nni.nas.pytorch.callbacks import LRSchedulerCallback
from nni.nas.pytorch.callbacks import ModelCheckpoint
from nni.algorithms.nas.pytorch.cream import CreamSupernetTrainer
from nni.algorithms.nas.pytorch.random import RandomMutator

def main():
    args, cfg = parse_config_args('nni.cream.supernet')

    # resolve logging
    output_dir = os.path.join(cfg.SAVE_PATH,
                              "{}-{}".format(datetime.date.today().strftime('%m%d'),
                                             cfg.MODEL))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # if args.local_rank == 0:
    #     logger = get_logger(os.path.join(output_dir, "train.log"))
    # else:
    #     logger = None

    # initialize distributed parameters
    torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # if args.local_rank == 0:
    #     logger.info(
    #         'Training on Process %d with %d GPUs.',
    #         args.local_rank, cfg.NUM_GPU)

    # fix random seeds
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # generate supernet
    model, sta_num, resolution = gen_supernet(
        flops_minimum=cfg.SUPERNET.FLOPS_MINIMUM,
        flops_maximum=cfg.SUPERNET.FLOPS_MAXIMUM,
        num_classes=cfg.DATASET.NUM_CLASSES,
        drop_rate=cfg.NET.DROPOUT_RATE,
        global_pool=cfg.NET.GP,
        resunit=cfg.SUPERNET.RESUNIT,
        dil_conv=cfg.SUPERNET.DIL_CONV,
        slice=cfg.SUPERNET.SLICE,
        verbose=cfg.VERBOSE,
        logger=None)

    # number of choice blocks in supernet
    # print(model.blocks)
    # choice_num = len(model.blocks[7])
    # if args.local_rank == 0:
    #     logger.info('Supernet created, param count: %d', (
    #         sum([m.numel() for m in model.parameters()])))
    #     logger.info('resolution: %d', (resolution))
        # logger.info('choice number: %d', (choice_num))

    # initialize flops look-up table
    model_est = FlopsEst(model)
    flops_dict, flops_fixed = model_est.flops_dict, model_est.flops_fixed
    for x in flops_dict.keys():
        print(flops_dict[x])
main()