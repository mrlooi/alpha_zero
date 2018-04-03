import copy
import numpy as np

from CoachTetris import Coach
from tetris.TetrisGame import TetrisGame as Game
from tetris.pytorch.NNetWrapper import NNetWrapper as nn
from utils import *

output_folder = './output'
args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': output_folder,
    'load_model': False,
    'load_folder_file': (output_folder,'checkpoint_7.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'train': dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'cuda': True, #torch.cuda.is_available(),
        'num_channels': 512,
    })
})

def main():
    n = 6
    m = 10
    g = Game(n, m)
    nnet = nn(g, args.train)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()

if __name__=="__main__":
    main()

    # n = 6
    # m = 10
    #
    # g = Game(n, m)
    # nnet = nn(g, args.train)
    #
    # c = Coach(g, nnet, args)
    # # c.learn()
    # self = c.mcts
    # board = c.game.getInitBoard()
    # # self.getActionProb(board, temp=1)
    # # # pi = c.mcts.getActionProb(board, temp=1)
    #