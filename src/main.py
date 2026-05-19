# coding: utf-8

"""
Main entry
# UPDATED
##########################
"""

import os
import argparse
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', '--gpuid', '--gpu-id', '-g', dest='gpu_id',
                        type=str, default=None, help='GPU id passed to CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1')

    args, _ = parser.parse_known_args()
    config_dict = {}
    if args.gpu_id is not None:
        config_dict['gpu_id'] = args.gpu_id

    from utils.quick_start import quick_start
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


