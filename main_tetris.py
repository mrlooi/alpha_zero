import copy
import numpy as np

from CoachTetris import Coach
from tetris.TetrisGame import TetrisGame as Game
from tetris.pytorch.NNetWrapper import NNetWrapper as nn
from utils import *

output_folder = '/home/vincent/hd/deep_learning/alpha-zero/tetris'
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
    

# class Coach():


#     """
#     This class executes the self-play + learning. It uses the functions defined
#     in Game and NeuralNet. args are specified in main.py.
#     """
#     def __init__(self, game, nnet, args):
#         self.game = game
#         self.args = args
        
#         self.nnet = nnet
#         self.pnet = self.nnet.__class__(self.game, self.args.train)  # the competitor network

#         self.mcts = MCTS(self.game, self.nnet, self.args)
#         self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
#         self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

#     def executeEpisode(self):
#         """
#         This function executes one episode of self-play, starting with player 1.
#         As the game is played, each turn is added as a training example to
#         trainExamples. The game is played till the game ends. After the game
#         ends, the outcome of the game is used to assign values to each example
#         in trainExamples.

#         It uses a temp=1 if episodeStep < tempThreshold, and thereafter
#         uses temp=0.

#         Returns:
#             trainExamples: a list of examples of the form (canonicalBoard,pi,v)
#                            pi is the MCTS informed policy vector, v is +1 if
#                            the player eventually won the game, else -1.
#         """
#         trainExamples = []
#         board = self.game.getInitBoard()

#         episodeStep = 0

#         while True:
#             episodeStep += 1
#             # canonicalBoard = self.game.getCanonicalForm(board)
#             temp = int(episodeStep < self.args.tempThreshold)

#             pi = self.mcts.getActionProb(board, temp=temp)
#             sym = self.game.getSymmetries(board, pi)
#             for b,p in sym:
#                 trainExamples.append([b, p, None])

#             action = np.random.choice(len(pi), p=pi)
            
#             board = self.game.getNextState(board, action)
            
#             r = self.game.getGameEnded(board)

#             if r!=0:
#                 return [(x[0],x[1],r) for x in trainExamples]

#     def learn(self):
#         from collections import deque
#         import time
#         from pytorch_classification.utils import Bar, AverageMeter

#         """
#         Performs numIters iterations with numEps episodes of self-play in each
#         iteration. After every iteration, it retrains neural network with
#         examples in trainExamples (which has a maximium length of maxlenofQueue).
#         It then pits the new neural network against the old one and accepts it
#         only if it wins >= updateThreshold fraction of games.
#         """

#         for i in range(1, self.args.numIters+1):
#             # bookkeeping
#             print('------ITER ' + str(i) + '------')
#             # examples of the iteration
#             if not self.skipFirstSelfPlay or i>1:
#                 iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
#                 eps_time = AverageMeter()
#                 bar = Bar('Self Play', max=self.args.numEps)
#                 end = time.time()
    
#                 for eps in range(self.args.numEps):
#                     self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
#                     iterationTrainExamples += self.executeEpisode()
    
#                     # bookkeeping + plot progress
#                     eps_time.update(time.time() - end)
#                     end = time.time()
#                     bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
#                                                                                                                total=bar.elapsed_td, eta=bar.eta_td)
#                     bar.next()
#                 bar.finish()

#                 # save the iteration examples to the history 
#                 self.trainExamplesHistory.append(iterationTrainExamples)
            

# class MCTS():
#     def __init__(self, game, nnet, args):
#         self.game = game
#         self.nnet = nnet
#         self.args = args
#         self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
#         self.Nsa = {}       # stores #times edge s,a was visited
#         self.Ns = {}        # stores #times board s was visited
#         self.Ps = {}        # stores initial policy (returned by neural net)

#         self.Es = {}        # stores game.getGameEnded ended for board s
#         self.Vs = {}        # stores game.getValidMoves for board s

#     def getActionProb(self, canonicalBoard, temp=1):
#         """
#         This function performs numMCTSSims simulations of MCTS starting from
#         canonicalBoard.

#         Returns:
#             probs: a policy vector where the probability of the ith action is
#                    proportional to Nsa[(s,a)]**(1./temp)
#         """
#         for i in range(self.args.numMCTSSims):
#             self.search(canonicalBoard)

#         s = self.game.stringRepresentation(canonicalBoard)
#         counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

#         if temp==0:
#             bestA = np.argmax(counts)
#             probs = [0]*len(counts)
#             probs[bestA]=1
#             return probs

#         counts = [x**(1./temp) for x in counts]
#         probs = [x/float(sum(counts)) for x in counts]
#         return probs

#     def search(self, canonicalBoard):

#         """
#         This function performs one iteration of MCTS. It is recursively called
#         till a leaf node is found. The action chosen at each node is one that
#         has the maximum upper confidence bound as in the paper.

#         Once a leaf node is found, the neural network is called to return an
#         initial policy P and a value v for the state. This value is propogated
#         up the search path. In case the leaf node is a terminal state, the
#         outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
#         updated.

#         Returns:
#             v: the negative of the value of the current canonicalBoard
#         """
#         canonicalBoard = copy.deepcopy(canonicalBoard)
        
#         s = self.game.stringRepresentation(canonicalBoard)
#         if s not in self.Es:
#             self.Es[s] = self.game.getGameEnded(canonicalBoard)
#         if self.Es[s]!=0:
#             # terminal node
#             return self.Es[s]

#         if s not in self.Ps:
#             # leaf node
#             self.Ps[s], v = self.nnet.predict(canonicalBoard.pieces)
#             valids = self.game.getValidMoves(canonicalBoard)
#             self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
#             sum_Ps_s = np.sum(self.Ps[s])
#             if sum_Ps_s > 0:
#                 self.Ps[s] /= sum_Ps_s    # renormalize
#             else:
#                 # if all valid moves were masked make all valid moves equally probable
                
#                 # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
#                 # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
#                 print("All valid moves were masked, do workaround.")
#                 self.Ps[s] = self.Ps[s] + valids
#                 self.Ps[s] /= np.sum(self.Ps[s])

#             self.Vs[s] = valids
#             self.Ns[s] = 0
#             return v

#         valids = self.Vs[s]
#         cur_best = -float('inf')
#         best_act = -1

#         # pick the action with the highest upper confidence bound
#         for a in range(self.game.getActionSize()):
#             if valids[a]:
#                 if (s,a) in self.Qsa:
#                     u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
#                 else:
#                     u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

#                 if u > cur_best:
#                     cur_best = u
#                     best_act = a

#         a = best_act
#         next_s = self.game.getNextState(canonicalBoard, a)
#         # next_s = self.game.getCanonicalForm(next_s)

#         v = self.search(next_s)

#         if (s,a) in self.Qsa:
#             self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
#             self.Nsa[(s,a)] += 1

#         else:
#             self.Qsa[(s,a)] = v
#             self.Nsa[(s,a)] = 1

#         self.Ns[s] += 1
#         return v

if __name__=="__main__":
    # main()

    n = 6
    m = 10
    
    g = Game(n, m)
    nnet = nn(g, args.train)

    c = Coach(g, nnet, args)
    # c.learn()
    self = c.mcts
    board = c.game.getInitBoard()
    self.getActionProb(board, temp=1)
    # # pi = c.mcts.getActionProb(board, temp=1)
