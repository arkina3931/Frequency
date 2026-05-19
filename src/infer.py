# coding: utf-8

"""
Load a saved best checkpoint and run full-sort inference on valid/test split.
"""

import argparse
import os
import sys

import torch

os.environ['NUMEXPR_MAX_THREADS'] = '48'

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_CWD = os.getcwd()
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)

from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.utils import dict2str, get_model, get_trainer, init_seed


def parse_topk(topk):
    if topk is None:
        return None
    values = [value.strip() for value in topk.split(',') if value.strip()]
    if not values:
        raise ValueError('--topk must contain at least one positive integer')
    return [int(value) for value in values]


def default_checkpoint_path(model, dataset):
    return os.path.join('saved', '{}-{}-best.pt'.format(model, dataset))


def resolve_checkpoint_path(checkpoint_path, default_path):
    raw_path = checkpoint_path or default_path
    if os.path.isabs(raw_path):
        return raw_path

    cwd_candidate = os.path.abspath(os.path.join(ORIGINAL_CWD, raw_path))
    src_candidate = os.path.abspath(os.path.join(SRC_DIR, raw_path))
    if checkpoint_path is None:
        return src_candidate
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    if os.path.exists(src_candidate):
        return src_candidate
    return cwd_candidate


def resolve_output_dir(output_dir):
    if output_dir is None or os.path.isabs(output_dir):
        return output_dir
    return os.path.abspath(os.path.join(ORIGINAL_CWD, output_dir))


def load_checkpoint(checkpoint_path):
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location='cpu')


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def get_saved_config(checkpoint):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get('config'), dict):
        return dict(checkpoint['config'])
    return {}


def materialize_search_values(config):
    hyper_parameters = config['hyper_parameters'] or []
    if 'seed' not in hyper_parameters:
        hyper_parameters = ['seed'] + hyper_parameters
    for key in hyper_parameters:
        value = config[key]
        if isinstance(value, list):
            config[key] = value[0] if value else None
    if config['seed'] is not None:
        init_seed(config['seed'])


def build_eval_data(config, split):
    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()
    eval_dataset = valid_dataset if split == 'valid' else test_dataset

    train_data = TrainDataLoader(
        config, train_dataset, batch_size=config['train_batch_size'], shuffle=True
    )
    train_data.pretrain_setup()
    eval_data = EvalDataLoader(
        config, eval_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']
    )
    return train_data, eval_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of model')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    parser.add_argument('--split', type=str, choices=['valid', 'test'], default='test', help='eval split')
    parser.add_argument('--gpu_id', '--gpuid', '--gpu-id', '-g', dest='gpu_id',
                        type=str, default=None, help='GPU id passed to CUDA_VISIBLE_DEVICES')
    parser.add_argument('--topk', type=str, default=None, help='comma-separated topk, e.g. 20 or 5,10,20,50')
    parser.add_argument('--recommend_dir', type=str, default=None, help='directory for saved top-k CSV')
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(
        args.checkpoint, default_checkpoint_path(args.model, args.dataset)
    )
    checkpoint = load_checkpoint(checkpoint_path)

    config_dict = get_saved_config(checkpoint)
    config_dict['checkpoint_file'] = checkpoint_path
    config_dict['save_recommended_topk'] = True
    if args.gpu_id is not None:
        config_dict['gpu_id'] = args.gpu_id
    topk = parse_topk(args.topk)
    if topk is not None:
        config_dict['topk'] = topk
    recommend_dir = resolve_output_dir(args.recommend_dir)
    if recommend_dir is not None:
        config_dict['recommend_topk'] = recommend_dir

    config = Config(args.model, args.dataset, config_dict)
    materialize_search_values(config)

    train_data, eval_data = build_eval_data(config, args.split)
    model = get_model(config['model'])(config, train_data).to(config['device'])
    model.load_state_dict(get_state_dict(checkpoint), strict=True)

    trainer = get_trainer()(config, model)
    result = trainer.evaluate(eval_data, is_test=True, idx='infer-{}'.format(args.split))

    print('Loaded checkpoint: {}'.format(checkpoint_path))
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print('Checkpoint epoch: {}'.format(checkpoint['epoch']))
    print('{} result: {}'.format(args.split, dict2str(result)))


if __name__ == '__main__':
    main()
