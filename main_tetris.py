import copy
import numpy as np
import os.path as osp

from CoachTetris import Coach
from tetris.TetrisGame import TetrisGame as Game
from tetris.pytorch.NNetWrapper import NNetWrapper as nn
from utils import *

n = 12
m = 15

output_folder = './models/%dx%dx%d'%(n,n,m)
args = dotdict({
    'numIters': 1000,
    'numEps': 40,
    'tempThreshold': 15,
    'updateThreshold': 0.6, 
    'maxlenOfQueue': 20000,
    'numMCTSSims': 30,
    'arenaCompare': 35,
    'cpuct': 1,

    'checkpoint': output_folder,
    'load_model': False,
    'load_folder_file': (output_folder,'checkpoint_4.pth.tar'),
    'load_examples': False,
    'examples_file': osp.join(output_folder,'data.example'),
    # 'numItersForTrainExamplesHistory': 4,
    # 'maxEpisodesInTrainHistory': 300,

    'train': dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 20,
        'batch_size': 32,
        'cuda': True, #torch.cuda.is_available(),
        'num_channels': 512,
    })
})

# def main():


if __name__=="__main__":
    g = Game(n, m)
    nnet = nn(g, args.train)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_examples:
        print("Load trainExamples from file")
        c.loadTrainExamples(examplesFile=args.examples_file, skipFirstSelfPlay=False)
    c.learn()