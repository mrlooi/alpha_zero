import numpy as np
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from collections import deque
import time
import glob

from pytorch_classification.utils import Bar, AverageMeter

from ArenaTetris import Arena
from MCTSTetris import MCTS

def get_latest_checkpoint_file(folder):
    ckpt_files = glob.glob(os.path.join(folder, "check*.pth.tar"))
    if len(ckpt_files) == 0:
        return None
    latest_ckpt_file = max(ckpt_files, key=os.path.getctime)
    latest_ckpt_file = latest_ckpt_file.split("/")[-1]
    return latest_ckpt_file

def get_latest_data_files(folder):
    data_files = glob.glob(os.path.join(folder, "*.example"))

    data_files = sorted(data_files, key=os.path.getctime, reverse=True)
    return data_files

class SelfPlay():

    def __init__(self, game, nnet, args):
        self.args = args
        self.game = game
        self.nnet = nnet
        self.mcts = None

        self.data = []

    def load_latest_checkpoint(self, folder):
        latest_ckpt = get_latest_checkpoint_file(folder)
        if latest_ckpt:
            self.nnet.load_checkpoint(folder, latest_ckpt)
        else:
            print("Did not find any checkpoint files in %s"%(folder))

    def run(self):
        folder = self.args.checkpoint

        print("Running self play...")

        while True:

            self.load_latest_checkpoint(folder)

            eps_time = AverageMeter()
            bar = Bar('Self Play', max=self.args.numEps)
            end = time.time()

            self.data = []
            for eps in range(self.args.numEps):
                self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree

                self.data.extend(self.executeEpisode())

                # bookkeeping + plot progress
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                           total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()
            bar.finish()

            # Save data
            self.saveTrainExamples(self.data, folder, "data_%d.example"%(time.time()*10))


    def saveTrainExamples(self, data, folder, filename):
        # folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, filename)
        with open(filename, "wb+") as f:
            print("Saving to %s..."%(filename))
            Pickler(f).dump(data)
            print("Saved to %s"%(filename))
        f.closed

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        # self.data = []
        data = []
        board = self.game.getInitBoard()

        episodeStep = 0

        while True:
            episodeStep += 1
            # canonicalBoard = self.game.getCanonicalForm(board)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(board, temp=temp)
            sym = self.game.getSymmetries(board, pi)
            for b,p in sym:
                data.append([b, p])

            action = np.random.choice(len(pi), p=pi)
            
            board = self.game.getNextState(board, action)
            
            has_ended = self.game.getGameEnded(board)

            if has_ended:
                r = self.game.getScore(board)
                return [(x[0],x[1],r) for x in data]

class Evaluator():

    def __init__(self, game, nnet, args):
        self.args = args
        self.game = game

        self.nnet = nnet

    def load_latest_checkpoint(self, folder):
        latest_ckpt = get_latest_checkpoint_file(folder)
        if latest_ckpt:
            self.nnet.load_checkpoint(folder, latest_ckpt)
        else:
            print("Did not find any checkpoint files in %s"%(folder))

    def run(self):
        folder = self.args.checkpoint
        best_model_file = 'best.pth.tar'

        update_threshold = self.args.updateThreshold


        best_net = self.nnet.__class__(self.game, self.args.train)

        while True:
            if not os.path.exists(os.path.join(folder, best_model_file)):
                sleep_secs = 30
                print("No best model '%s' found in '%s' folder, sleeping for %d seconds"%(best_model_file, folder, sleep_secs))
                time.sleep(sleep_secs)
            else:
                best_net.load_checkpoint(folder=folder, filename=best_model_file)
                break

        while True:
            self.load_latest_checkpoint(folder)
            
            pmcts = MCTS(self.game, best_net, self.args)            
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST BEST VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/BEST WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < update_threshold:
                print('FAILED TO BEAT BEST MODEL ABOVE THRESHOLD OF %.3f'%(update_threshold))
                # self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('SAVING TO BEST MODEL...')
                self.nnet.save_checkpoint(folder=folder, filename=best_model_file)           
                best_net.load_checkpoint(folder=folder, filename=best_model_file)
                print("Best net updated")

class Trainer():
    def __init__(self, game, nnet, args):
        self.args = args
        self.game = game

        self.nnet = nnet

        max_examples = self.args.maxlenOfQueue # maxEpisodesInTrainHistory
        self.trainExamplesHistory = deque([], max_examples)

        self.folder = self. args.checkpoint

        self.cur_data_files = []

    def load_latest_data(self, folder):

        max_examples = self.args.maxlenOfQueue # maxEpisodesInTrainHistory

        data_files = get_latest_data_files(folder)

        new_data_files = list(np.setdiff1d(data_files, self.cur_data_files))
        new_data_files = sorted(new_data_files, key=os.path.getctime, reverse=True)
        new_data_files_copy = list(new_data_files)
        for f in new_data_files:
            try:
                data = self._load_data_file(f)

                self.trainExamplesHistory.extendleft(data)
                if len(self.trainExamplesHistory) == max_examples:
                    break
            except EOFError, e:
                print("EOF error on file %s, skipping"%(f))
                new_data_files_copy.remove(f)
            except ValueError, e:
                print("Value error on file %s, skipping"%(f))
                new_data_files_copy.remove(f)

        self.cur_data_files.extend(new_data_files_copy)
        return len(new_data_files_copy) > 0  # has new data file(s)

    def _load_data_file(self, file):
        print("Reading %s..."%(file))
        with open(file, "rb") as f:
            data = Unpickler(f).load()
        f.closed
        print("Loaded %s"%(file))

        return data

    def run(self):
        folder = self.folder

        min_examples = self.args.minlenOfQueue # maxEpisodesInTrainHistory

        iter_ = 0
        while True:

            has_new_data = self.load_latest_data(folder)
            if not has_new_data:
                sleep_secs = 30
                print("No new data found in '%s' folder, sleeping for %d seconds"%(folder, sleep_secs))
                time.sleep(sleep_secs)
                continue

            iter_ += 1
            print("TRAIN ITER %d"%(iter_))

            total_examples = len(self.trainExamplesHistory)

            print("Current data length %d"%(total_examples))
            if total_examples < min_examples:
                print("%d samples has not exceeded minimum threshold of %d, skipping training..."%(total_examples, min_examples))
                continue

            batch_sz = self.args.train.batch_size
            if total_examples < batch_sz:
                print("Total available examples %d < batch size of %d, skipping training/pitting.."%(total_examples, batch_sz))
                continue
            
            self.nnet.train(self.trainExamplesHistory)

            # save 
            self.nnet.save_checkpoint(folder=folder, filename='checkpoint_%d.pth.tar'%(time.time()*10))


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.args = args
        
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, self.args.train)  # the competitor network
        self.mcts = MCTS(self.game, self.nnet, self.args)

        self.trainExamplesHistory = [] # deque([], maxlen=self.args.maxlenOfQueue)    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()

        episodeStep = 0

        while True:
            episodeStep += 1
            # canonicalBoard = self.game.getCanonicalForm(board)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(board, temp=temp)
            sym = self.game.getSymmetries(board, pi)
            for b,p in sym:
                trainExamples.append([b, p])

            action = np.random.choice(len(pi), p=pi)
            
            board = self.game.getNextState(board, action)
            
            has_ended = self.game.getGameEnded(board)

            if has_ended:
                r = self.game.getScore(board)
                return [(x[0],x[1],r) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        max_examples = self.args.maxlenOfQueue # maxEpisodesInTrainHistory
        min_examples = self.args.minlenOfQueue # maxEpisodesInTrainHistory

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                # iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
    
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
    
                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    episode_data = self.executeEpisode()
                    self.trainExamplesHistory.extend(episode_data) 
    
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # # save the iteration examples to the history 
                # self.trainExamplesHistory.append(iterationTrainExamples)
            
            total_examples = len(self.trainExamplesHistory)

            print("Current data length %d"%(total_examples))
            if total_examples < min_examples:
                print("%d samples has not exceeded minimum threshold of %d, skipping training..."%(total_examples, min_examples))
                continue
            if total_examples > max_examples:
                print("len(trainExamplesHistory) = %d => keep the latest trainExamples"%(total_examples))
                # self.trainExamplesHistory.pop(0)
                self.trainExamplesHistory = self.trainExamplesHistory[-max_examples:]
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            if not self.skipFirstSelfPlay or i>1:
                self.saveTrainExamples(self.args.checkpoint, "data.example")
            
            # shuffle examlpes before training
            # trainExamples = self.trainExamplesHistory
            # for e in self.trainExamplesHistory:
            #     trainExamples.extend(e)
            # shuffle(trainExamples)

            batch_sz = self.args.train.batch_size
            if total_examples < batch_sz:
                print("Total available examples %d < batch size of %d, skipping training/pitting.."%(total_examples, batch_sz))
                continue

            # training new network, keeping a copy of the old one
            if not os.path.exists(os.path.join(self.args.checkpoint, 'best.pth.tar')):
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                continue
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            
            pmcts = MCTS(self.game, self.pnet, self.args)
            
            self.nnet.train(self.trainExamplesHistory)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST BEST VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                # self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('SAVING TO BEST MODEL...')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            print("Saved to %s"%(os.path.join(self.args.checkpoint, self.getCheckpointFile(i))))

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, folder, filename):
        # folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, filename)
        with open(filename, "wb+") as f:
            print("Saving to %s..."%(filename))
            Pickler(f).dump(self.trainExamplesHistory)
            print("Saved to %s"%(filename))
        f.closed

    def loadTrainExamples(self, examplesFile, skipFirstSelfPlay=True):
        # examplesFile = os.path.join(self.args.load_folder_file[0], "checkpoint_0.pth.tar.examples")
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("Could not find %s file. Continue? [y|n]"%(examplesFile))
            if r != "y":
                sys.exit()
        else:
            print("Found trainExamples file: %s. Reading it..."%(examplesFile))
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = skipFirstSelfPlay
