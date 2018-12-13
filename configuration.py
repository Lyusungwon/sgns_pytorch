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
    model_arg.add_argument('--embed-size', default=300, type=int)

    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--data-dir', default='/home/sungwon/data/corpus', type=str, help='directory of training/testing data (default: datasets)')
    data_arg.add_argument('--dataset', default='the_lord_of_the_rings.txt', type=str)
    data_arg.add_argument('--window-size', default=5, type=int)
    data_arg.add_argument('--neg-sample-size', default=7, type=int)
    data_arg.add_argument('--remove-th', default=3, type=int)
    data_arg.add_argument('--subsample-th', default=1e-4, type=float)
 
    train_arg = parser.add_argument_group('Train')
    train_arg.add_argument('--device', default=0, type=int)
    train_arg.add_argument('--batch-size', default=64, type=int, help='mini-batch size (default: 64)')
    train_arg.add_argument('--epochs', default=128, type=int, help='number of total epochs (default: 10)')
    train_arg.add_argument('--lr', default=0.025, type=float, help='learning rate (default: 0.0002)')
    train_arg.add_argument('--log-interval', default=100, type=int)
    train_arg.add_argument('--save-interval', default=10, type=int)
    train_arg.add_argument('--timestamp', default=datetime.now().strftime("%y%m%d%H%M%S"), type=str)
    train_arg.add_argument('--load-model', default=None, type=str)
    train_arg.add_argument('--log-dir', default='saved/runs/', type=str)
    train_arg.add_argument('--multigpu', action='store_true')

    #for large dataset dataloader
    train_arg.add_argument('--multi-node', action='store_true')
    train_arg.add_argument('--multi-gpu', action='store_true')
    train_arg.add_argument('--num-workers', default=0, type=int)
    train_arg.add_argument('--backend', default='tcp', type=str)
    train_arg.add_argument('--init-method', default='nccl://127.0.0.1:22', type=str)
    train_arg.add_argument('--rank', type=int)
    train_arg.add_argument('--world-size', type=int)
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config_list = [args.model_name, args.embed_size, \
                   args.dataset, args.window_size, args.neg_sample_size,\
                   args.batch_size, args.epochs, args.lr, args.device]
    args.config = '_'.join(list(map(str, config_list))).replace("/", ".")
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.log_dir + args.load_model, map_location=lambda storage,loc: storage))
        args.timestamp = args.load_model_code[:12]
        print('Model loaded')
    return args
