import argparse
import models
import time
from tqdm import tqdm
import sys, os
import warnings
from pathlib import Path
import torch
import numpy as np
from scipy.stats import norm
from scipy.io.wavfile import write
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import scipy
import librosa
import torch.nn as nn
import matplotlib.pyplot as plt

def parse_args(parser):
    parser.add_argument('-o', '--output', type=str, required=False, default='models/',
                        help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',
                        help='Path to dataset')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=200,
                          help='Number of total epochs to run')
    training.add_argument('--enable_cuda', default=True)
    training.add_argument('--device', default='cpu')
    training.add_argument('--epochs-per-checkpoint', type=int, default=50,
                          help='Number of epochs per checkpoint')
    training.add_argument('--checkpoint-path', type=str, default=None,
                          help='Checkpoint path to resume training')
    training.add_argument('--resume', action='store_true',
                          help='Resume training from the last available checkpoint')
    training.add_argument('--seed', type=int, default=1234,
                          help='Seed for PyTorch random number generators')
    training.add_argument('--amp', action='store_true',
                          help='Enable AMP')
    training.add_argument('--cuda', action='store_true',
                          help='Run on GPU using CUDA')
    training.add_argument('--cudnn-benchmark', action='store_true',
                          help='Enable cudnn benchmark mode')
    training.add_argument('--ema-decay', type=float, default=0,
                          help='Discounting factor for training weights EMA')
    training.add_argument('--gradient-accumulation-steps', type=int, default=1,
                          help='Training steps to accumulate gradients for')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument('--optimizer', type=str, default='adam',
                              help='Optimization algorithm')
    optimization.add_argument('-lr', '--learning-rate', type=float, default=0.0001,
                              help='Learing rate')
    optimization.add_argument('--weight-decay', default=1e-6, type=float,
                              help='Weight decay')
    optimization.add_argument('--grad-clip-thresh', default=1000.0, type=float,
                              help='Clip threshold for gradients')
    optimization.add_argument('-bs', '--h-size', type=int, default=4,
                              help='Batch size per GPU')
    optimization.add_argument('--warmup-steps', type=int, default=1000,
                              help='Number of steps for lr warmup')
    optimization.add_argument('--dur-predictor-loss-scale', type=float,
                              default=1.0, help='Rescale duration predictor loss')
    optimization.add_argument('--pitch-predictor-loss-scale', type=float,
                              default=1.0, help='Rescale pitch predictor loss')

    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument('--training-files', type=str, default=None,
                         help='Path to training filelist. Separate multiple paths with commas.')
    dataset.add_argument('--validation-files', type=str, default=None,
                         help='Path to validation filelist. Separate multiple paths with commas.')
    dataset.add_argument('--pitch-mean-std-file', type=str, default=None,
                         help='Path to pitch stats to be stored in the model')
    dataset.add_argument('--text-cleaners', nargs='*',
                         default=['english_cleaners'], type=str,
                         help='Type of text cleaners for input text')
    dataset.add_argument('--symbol-set', type=str, default='english_basic',
                         help='Define symbol set for input text')

    cond = parser.add_argument_group('conditioning on additional attributes')
    cond.add_argument('--n-speakers', type=int, default=1,
                      help='Condition on speaker, value > 1 enables trainable speaker embeddings.')

    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0),
                             help='Rank of the process for multiproc. Do not set manually.')
    distributed.add_argument('--world_size', type=int, default=os.getenv('WORLD_SIZE', 1),
                             help='Number of processes for multiproc. Do not set manually.')
    return parser

def predict(m_t, device, checkpoint):

    parser = argparse.ArgumentParser(description='PyTorch FastPitch Training',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    args.device = device
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    desc = 'FastPitch_aai'
    parser = models.parse_model_args(desc, parser)
    args, unk_args = parser.parse_known_args()
    model_config = models.get_model_config(desc, args)
    net = models.get_model(desc, model_config, args.device)
    if device != 'cpu':
        net = net.float()
    net.load_state_dict(torch.load(checkpoint))
    m_t = torch.from_numpy(m_t).unsqueeze(dim=0)
    net.eval()
    return net(m_t.to(args.device), None)[0].squeeze().detach().cpu().numpy()
