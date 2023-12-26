# Based on https://github.com/aimagelab/mammoth

from readline import set_pre_input_hook
import numpy # needed (don't change it)
import importlib
import os
import sys
import socket
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(mammoth_path)
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/model')

from datasets import NAMES as DATASET_NAMES
from model import get_all_models, get_model
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from datasets import get_dataset
from utils.training import ctrain
from utils.conf import set_random_seed
import setproctitle
import torch
import uuid
import datetime

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
   
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('model.' + args.model)
    

    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args

def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()    

    #Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)
    
    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('model.' + args.model), 'Buffer') and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()
    
    # Get model architecture
    
    backbone = dataset.get_backbone()
    
    loss = dataset.get_loss()
    
    
    model = get_model(args, backbone, loss, dataset.get_transform())
    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))     
    if isinstance(dataset, ContinualDataset):
        ctrain(model, dataset, args)
        
if __name__ == '__main__':
    main()
