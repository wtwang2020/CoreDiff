import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.distributed as dist
import argparse
import sys
sys.path.append('/data/wangweitao/i_am_yibo/mnt/sdb/i_am_yibo/project/欧核/主模型/OTF_nii')

from models import model_dict, TrainTask

if __name__ == '__main__':
    # reference https://stackoverflow.com/questions/38050873/can-two-python-argparse-objects-be-combined/38053253
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args()
    MODEL = model_dict[default_opt.model_name]
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    # dist.init_process_group(backend='nccl',
    #                         init_method='env://')
    # torch.cuda.set_device(dist.get_rank())
    model = MODEL(opt)
    model.fit()
