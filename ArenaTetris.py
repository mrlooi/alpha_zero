import numpy as np
import copy 
import time

from pytorch_classification.utils import Bar, AverageMeter

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, player1_is_human=False, player2_is_human=False, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.player1_is_human = player1_is_human
        self.player2_is_human = player2_is_human
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        player_scores = [-1,-1]
        players = [self.player1, self.player2]
        is_human = [self.player1_is_human, self.player2_is_human]
        
        original_board = self.game.getInitBoard()

        final_boards = []

        for ix in xrange(2): # 2 players
            board = copy.deepcopy(original_board)
            it = 0
            while not self.game.getGameEnded(board):
                it+=1

                p = players[ix]
                action = p(board) # Action here!
                if is_human[ix]:
                    break

                if verbose:
                    print("Player %d, Turn %d"%(ix + 1, it))
                if self.display:
                    self.display(board)

                valids = self.game.getValidMoves(board)

                if valids[action]==0: # BAD
                    print(action)
                    assert valids[action] > 0
                board = self.game.getNextState(board, action)

            score = self.game.getScore(board)
            assert(score is not False)
            player_scores[ix] = score
            print("Finished Game -> Player %d score: %.3f"%(ix, score))

            final_boards.append(board)

        if verbose:
            print("Game over -> Player 1 Score: %.3f, Player 2 Score: %.3f"%(player_scores[0], player_scores[1]))

        if self.display:
            p1_board = final_boards[0]
            p2_board = final_boards[1]
            self.display(p1_board, "Player1")
            self.display(p2_board, "Player2")

        return player_scores[0], player_scores[1]

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        # num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0

        player_score_history = [[],[]]  # p1, p2 histories
        for _ in range(num):
            p1_score, p2_score = self.playGame(verbose=verbose)
            player_score_history[0].append(p1_score)
            player_score_history[1].append(p2_score)

            if p1_score > p2_score:
                oneWon+=1
            elif p2_score > p1_score:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        # self.player1, self.player2 = self.player2, self.player1
        
        # for _ in range(num):
        #     gameResult = self.playGame(verbose=verbose)
        #     if gameResult==-1:
        #         oneWon+=1                
        #     elif gameResult==1:
        #         twoWon+=1
        #     else:
        #         draws+=1
        #     # bookkeeping + plot progress
        #     eps += 1
        #     eps_time.update(time.time() - end)
        #     end = time.time()
        #     bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num, et=eps_time.avg,
        #                                                                                                total=bar.elapsed_td, eta=bar.eta_td)
        #     bar.next()
            
        bar.finish()

        player_score_history = np.array(player_score_history)
        player_mean_scores = np.mean(player_score_history, axis=1)
        print("Total Games: %d, Average Scores -> P1: %.3f, P2: %.3f"%(num, player_mean_scores[0], player_mean_scores[1]))

        return oneWon, twoWon, draws


if __name__=="__main__":
    import cv2
    import random

    from tetris.TetrisPlayers import RandomPlayer
    from tetris.pytorch.NNetWrapper import NNetWrapper as NNet
    from MCTSTetris import MCTS
    

    USE_V2 = True
    if USE_V2:
        from tetris.TetrisLogic2 import BoardRenderer2
        from tetris.TetrisGame2 import TetrisGame2 as Game2
        
        n = 10
        r = 6
        c = 6
        g = Game2(r,c,n)
        b_renderer = BoardRenderer2(unit_res=30)
        model_folder = './models/%dx%dx%d_v2'%(r,c,n)

    else:
        from tetris.TetrisLogic import BoardRenderer
        from tetris.TetrisGame import TetrisGame as Game

        n = 12
        m = 15
        g = Game(n,m)
        b_renderer = BoardRenderer(unit_res=30)

        model_folder = './models/%dx%dx%d'%(n,n,m)


    def display_func(board_obj, title="board_img"):
        board_img = b_renderer.display_board(board_obj)
        cv2.imshow(title, board_img)
        cv2.waitKey(0)

    def random_vs_random(g):
        player1 = RandomPlayer(g)
        player2 = RandomPlayer(g)

        arena = Arena(player1.play, player2.play, g, display=None)#display_func)
        pwins, nwins, draws = arena.playGames(40, verbose=False)
        print(pwins, nwins, draws)

    def nnet_vs_random(nnet_mcts, g):

        n1p = lambda x: np.argmax(nnet_mcts.getActionProb(x, temp=0))

        r_player = RandomPlayer(g)

        arena = Arena(n1p, r_player.play, g, display=None) #display_func)
        print(arena.playGames(20, verbose=True))

    def nnet_vs_human(nnet_mcts, g):
        USE_V2 = True
        if USE_V2:
            from tetris.PlayerGui2 import PlayerGUI2 as PlayerGUI
        else:
            from tetris.PlayerGui import PlayerGUI

        n1p = lambda x: np.argmax(nnet_mcts.getActionProb(x, temp=0))
        player = PlayerGUI("PlayerOne", g, b_renderer)

        arena = Arena(player.play, n1p, g, player1_is_human=True, display=display_func)
        print(arena.playGames(20, verbose=True))

    # random_vs_random(g)
    # import sys
    # sys.exit(0)

    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

    # nnet players vs random
    nnet_args = dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 15,
        'batch_size': 64,
        'cuda': True, #torch.cuda.is_available(),
        'num_channels': 512,
    })

    n1 = NNet(g, nnet_args)
    n1.load_checkpoint(model_folder, 'best.pth.tar')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)

    # random_vs_random(g)
    nnet_vs_random(mcts1, g)
    # nnet_vs_human(mcts1, g)
