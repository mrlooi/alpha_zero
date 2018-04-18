import copy
import numpy as np
import os.path as osp
from utils import dotdict

USE_VOXEL = True
if USE_VOXEL:
    from voxel.VoxelRender import BoardRenderer
    from voxel.VoxelGame import VoxelGame
    
    n = 20
    x = 9 
    y = 4
    z = 6
    g = VoxelGame(x, y, z, n)
    output_folder = './models2/voxel/%dx%dx%d_%d'%(x,y,z,n)

    from voxel.pytorch.NNetWrapper import NNetWrapper as nn

else:
    USE_V2 = True
    if USE_V2:
        from tetris.TetrisGame2 import TetrisGame2 as Game2
        
        n = 15
        r = 8
        c = 10
        g = Game2(r,c,n)

        output_folder = './models/%dx%dx%d_v2'%(r,c,n)
    else:
        from tetris.TetrisGame import TetrisGame as Game
        n = 6
        m = 10
        g = Game(n, m)

        output_folder = './models/%dx%dx%d'%(n,n,m)

    from tetris.pytorch.NNetWrapper import NNetWrapper as nn


args = dotdict({
    'numIters': 20,
    'numEps': 50,
    'tempThreshold': 15,
    'updateThreshold': 0.6, 
    'minlenOfQueue': 3000,
    'maxlenOfQueue': 8000,
    'numMCTSSims': 30,
    'arenaCompare': 30,
    'cpuct': 1,

    'checkpoint': output_folder,
    'load_model': False,
    'load_folder_file': (output_folder,'temp.pth.tar'),
    'load_examples': True,
    'examples_file': osp.join(output_folder,'data.example'),
    # 'numItersForTrainExamplesHistory': 4,
    # 'maxEpisodesInTrainHistory': 300,

    'train': dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 30,
        'batch_size': 64,
        'cuda': True, #torch.cuda.is_available(),
        'num_channels': 512,
    })
})

CMD_LIST = ['self', 'opt', 'eval']

def create_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="what to do", choices=CMD_LIST)
    return parser

if __name__=="__main__":
    parser = create_parser()
    p_args = parser.parse_args()

    nnet = nn(g, args.train)

    if p_args.cmd == "self":
        from CoachTetris import SelfPlay
        worker = SelfPlay(g, nnet, args)
    elif p_args.cmd == 'opt':
        from CoachTetris import Trainer

        if args.load_model:
            nnet.load_checkpoint(args.checkpoint, args.load_folder_file[1])

        worker = Trainer(g, nnet, args)
    elif p_args.cmd == 'eval':
        from CoachTetris import Evaluator
        worker = Evaluator(g, nnet, args)
    else:
        raise ValueError("Command '%s' does not exist! Available commands: %s"%(p_args.cmd, CMD_LIST))

    worker.run()

    # c = Coach(g, nnet, args)
    # if args.load_examples:
    #     print("Load trainExamples from file")
    #     c.loadTrainExamples(examplesFile=args.examples_file, skipFirstSelfPlay=True)
    # c.learn()