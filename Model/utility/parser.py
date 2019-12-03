'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import sys

import argparse
import copy

import datetime
from pytz import timezone
import getpass

import pprint as pp
import pathlib
from pathlib import Path

from utility.gc_storage import GCStorage
from utility.logger import Logger
from utility.constants import *


def str_to_bool(arg):
    """Convert an argument string into its boolean value.

    Args:
        arg (string): String representing a bool.

    Returns:
        (bool) Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_experiment_number(experiments_dir, experiment_name):
    """Parse directory to count the previous copies of an experiment."""
    dir_structure = GCStorage.MONO.list_files(experiments_dir)
    dirnames = [exp_dir.split('/')[-1] for exp_dir in dir_structure[1]]

    ret = 1
    for d in dirnames:
        if d[:d.rfind('_')] == experiment_name:
            ret = max(ret, int(d[d.rfind('_') + 1:]) + 1)
    return ret

def namespace_to_dict(args):
    """Turn a nested Namespace object into a nested dictionary."""
    args_dict = vars(copy.deepcopy(args))

    for arg in args_dict:
        obj = args_dict[arg]
        if isinstance(obj, argparse.Namespace):
            item = namespace_to_dict(obj)
            args_dict[arg] = item
        else:
            if isinstance(obj, pathlib.PosixPath):
                args_dict[arg] = str(obj)

    return args_dict

def wrap_up(args):
    
    cloudFS = GCStorage.get_CloudFS(PROJECT_ID, GC_BUCKET, CREDENTIAL_PATH)

    us_timezone = timezone('US/Pacific')
    date = datetime.datetime.now(us_timezone).strftime("%Y-%m-%d")
    save_dir = Path(EXP_STORAGE) / date

    args.exp_name = getpass.getuser() + '_KGAT_' + args.exp_name + '_' + args.dataset
    exp_num = get_experiment_number(save_dir, args.exp_name)

    args.exp_name = args.exp_name + '_' + str(exp_num)
    save_dir = save_dir / args.exp_name
    log_file = save_dir / 'run_log.txt'

    arg_dict = namespace_to_dict(args)
    arg_dict_text = pp.pformat(arg_dict, indent=4)

    arg_text = ' '.join(sys.argv)

    args.logger = Logger(log_file, save_dir)

    args.logger.log_text({'setup:command_line': arg_text,
                          'setup:parsed_arguments': arg_dict_text},
                         0, False)

    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Run KGAT.")

    parser.add_argument('--exp_name', default='KGAT',
                        help='experiment_name')

    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='yelp2018',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='CF Embedding size.')
    parser.add_argument('--kge_size', type=int, default=64,
                        help='KG Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='CF batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=2048,
                        help='KG batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='kgat',
                        help='Specify a loss type from {kgat, bprmf, fm, nfm, cke, cfkg}.')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--alg_type', nargs='?', default='ngcf',
                        help='Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--use_att', type=bool, default=True,
                        help='whether using attention mechanism')
    parser.add_argument('--use_kge', type=bool, default=True,
                        help='whether using knowledge graph embedding')

    parser.add_argument('--use_skip', type=str_to_bool, default=False,
                        help='Whether to hop twice for neighbor definition')



    args = parser.parse_args()

    args = wrap_up(args)

    return args