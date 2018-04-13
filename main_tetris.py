import copy
import numpy as np
import os.path as osp
from utils import dotdict

USE_V2 = True
if USE_V2:
    from tetris.TetrisGame2 import TetrisGame2 as Game2
    
    n = 15
    r = 10
    c = 10
    g = Game2(r,c,n)

    output_folder = './models/%dx%dx%d_v2'%(r,c,n)
else:
    from tetris.TetrisGame import TetrisGame as Game
    n = 6
    m = 10
    g = Game(n, m)

    output_folder = './models/%dx%dx%d'%(n,n,m)

args = dotdict({
    'numIters': 10,
    'numEps': 80,
    'tempThreshold': 15,
    'updateThreshold': 0.6, 
    'maxlenOfQueue': 6000,
    'numMCTSSims': 40,
    'arenaCompare': 30,
    'cpuct': 1,

    'checkpoint': output_folder,
    'load_model': False,
    'load_folder_file': (output_folder,'temp.pth.tar'),
    'load_examples': False,
    'examples_file': osp.join(output_folder,'data.example'),
    # 'numItersForTrainExamplesHistory': 4,
    # 'maxEpisodesInTrainHistory': 300,

    'train': dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 15,
        'batch_size': 64,
        'cuda': True, #torch.cuda.is_available(),
        'num_channels': 512,
    })
})

# def main():


if __name__=="__main__":

    from CoachTetris import Coach
    from tetris.pytorch.NNetWrapper import NNetWrapper as nn

    nnet = nn(g, args.train)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_examples:
        print("Load trainExamples from file")
        c.loadTrainExamples(examplesFile=args.examples_file, skipFirstSelfPlay=False)
    c.learn()