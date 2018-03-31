from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
# from .TetrisLogic import Board
from TetrisLogic import Board
import numpy as np
import random


class TetrisGame(Game):
    def __init__(self, n):
        self.n = n

        self.min_width=1
        self.max_width=int(self.n/2) + 1
        self.min_height=1
        self.max_height=int(self.n/3)

        self.box_list = self.getInitBoxList()
        self.action_list, self.box_action_dict = self.getActionList(self.box_list)

        self.action_sz = len(self.action_list)

    def getActionList(self, box_list):
        board_sz = self.n * self.n
        b = Board(self.n)
        action_list = []
        rev_dict = {}

        idx = 0
        for bx_l in box_list:
            # action_list.extend([[a, bx_l] for a in xrange(board_sz) if b.is_valid_placement(self.boardIndexToSquare(a), bx_l)])
            # rev_dict[bx_l] = []
            for a in xrange(board_sz):
                # if b.is_valid_placement(self.boardIndexToSquare(a), bx_l):
                action_list.append([a, bx_l])
                # rev_dict[bx_l].append([a, idx])
                x, y = self.boardIndexToSquare(a)
                k = tuple((bx_l[0],bx_l[1],x,y))  # w h x y
                rev_dict[k] = idx
                idx += 1
        return action_list, rev_dict

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return b.pieces

    def getInitBoxList(self):
        # sort by width
        box_list = []
        for w in xrange(self.min_width, self.max_width+1):
            for h in xrange(self.min_height, self.max_height+1):
                box_list.append((w,h))
        return box_list

    def generateRandomBoxList(self):
        boxes = []
        total_cells = self.n * self.n
        acc_cells = 0
        while acc_cells < total_cells:
            w = random.randint(self.min_width, self.max_width)
            h = random.randint(self.min_height, self.max_height)
            acc_cells += w * h
            boxes.append((w,h))
        return boxes

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        # return self.n*self.n + 1
        # box_list_cnt = len(getInitBoxList())
        # board_sz = np.size(getInitBoard())
        # return board_sz * box_list_cnt + 1
        return self.action_sz

    def boardIndexToSquare(self, idx):
        x = int(idx%self.n)
        y = int(idx/self.n) 
        return x,y

    def getNextState(self, board, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return board
        b = Board(self.n)
        b.pieces = np.copy(board)
        square_idx, box_size = self.action_list[action]
        move = self.boardIndexToSquare(square_idx)
        b.execute_move(move, box_size)
        return b.pieces

    def getValidMoves(self, board, box_list):
        # return a fixed size binary vector
        valids = [0] * self.action_sz
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = []

        for box_size in box_list:
            b_legal_moves = b.get_legal_moves(box_size)
            w,h = box_size
            for lm in b_legal_moves:
                x,y = lm
                try:
                    legalMoves.append(self.box_action_dict[(w,h,x,y)])
                except KeyError, e:
                    print(e)

        if len(legalMoves)==0:
            valids[-1]=1
        else:
            for ix in legalMoves:
                valids[ix]=1
        return np.array(valids)

    def getGameEnded(self, board, remaining_box_list):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.is_full():
            return 1
        for box_size in remaining_box_list:
            if b.has_legal_moves(box_size):
                return 0
        # if b.has_legal_moves(-player):
        #     return 0
        # if b.countDiff(player) > 0:
        #     return 1
        # return -1
        # return 1
        return self.getScore(b)

    def getScore(self, board):
        b = Board(self.n)
        b.pieces = np.copy(board)
        # return b.count_occupied()
        return b.get_score()

    def getCanonicalForm(self, board):
        # return state if player==1, else return -state if player==-1
        return board

    # TODO: CHECK WTF is 'pi'
    def getSymmetries(self, board, pi):
        # horizontal flip only 
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        # for i in range(1, 5):
        for j in [True, False]:
            newB = board # np.rot90(board, i)
            newPi = pi_board # np.rot90(pi_board, i)
            if j:
                newB = np.fliplr(newB)
                newPi = np.fliplr(newPi)
            l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

def display(board):
    n = board.shape[0]

    for y in range(n):
        print (y,"|",end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|",end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1: print("b ",end="")
            elif piece == 1: print("W ",end="")
            else:
                if x==n:
                    print("-",end="")
                else:
                    print("- ",end="")
        print("|")

    print("   -----------------------")


if __name__ == '__main__':
    # CREATE VIEWER OF VALID MOVES AND NEXT STATE
    n = 6
    g = TetrisGame(n)
    b = Board(n)

    rand_box_list = g.generateRandomBoxList()
    total_actions = g.getActionSize()

    for box_size in rand_box_list:
        print(box_size)
        valid_moves = g.getValidMoves(b.pieces, [box_size])
        # pick random valid move
        valid_move_idx = np.where(valid_moves==1)[0]
        rand_move = random.choice(valid_move_idx)
        b.pieces = g.getNextState(b.pieces, rand_move)
        break
