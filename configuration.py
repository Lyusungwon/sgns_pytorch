import os
import argparse
import torch
from datetime import datetime
from pathlib import Path
import re
home = str(Path.home())


def get_config():
    parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--model-name', default='sgns', type=str)
    model_arg.add_argument('--embed-size', default=200, type=int)

    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data-dir', default='./data', type=str, help='directory of training/testing data (default: datasets)')
    data_arg.add_argument('--dataset', default='harry_potter.txt', type=str)
    data_arg.add_argument('--window-size', default=5, type=int)
    data_arg.add_argument('--neg-sample-size', default=10, type=int)
    data_arg.add_argument('--remove-th', default=5, type=int)
    data_arg.add_argument('--subsample-th', default=2e-3, type=float)
 
    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--device', default=0, type=int)
    train_arg.add_argument('--batch-size', default=1024, type=int, help='mini-batch size (default: 64)')
    train_arg.add_argument('--epochs', default=256, type=int, help='number of total epochs (default: 10)')
    train_arg.add_argument('--lr', default=1e-5, type=float, help='learning rate (default: 0.0002)')
    train_arg.add_argument('--log-interval', default=100, type=int)
    train_arg.add_argument('--save-interval', default=64, type=int)
    train_arg.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M%S"), type=str)
    train_arg.add_argument('--load-model', default=None, type=str)
    train_arg.add_argument('--log-dir', default='saved/runs/', type=str)

    #for large dataset dataloader
    train_arg.add_argument('--multi-gpu', action='store_true')
    train_arg.add_argument('--num-gpu', default=1, type=int)
    train_arg.add_argument('--multi-node', action='store_true')
    train_arg.add_argument('--num-workers', default=20, type=int)
    train_arg.add_argument('--backend', default='nccl', type=str)
    train_arg.add_argument('--init-method', default='nccl://deepspark.snu.ac.kr', type=str)
    train_arg.add_argument('--rank', type=int)
    train_arg.add_argument('--world-size', type=int)
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')
    config_list = [args.model_name, args.embed_size, \
                   args.dataset, args.window_size, args.neg_sample_size,\
                   args.batch_size, args.epochs, args.lr, args.multi_gpu, args.num_gpu]
    args.config = '_'.join(list(map(str, config_list))).replace("/", ".")
    if args.load_model is not None:
        args.timestamp = args.load_model[:12]
        print('Model loaded')
    args.log_dir = os.path.join(args.log_dir, args.timestamp + '_' + args.config)
    return args
