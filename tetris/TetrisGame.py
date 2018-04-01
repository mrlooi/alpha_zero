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


class BoardRenderer(object):

    def __init__(self, unit_res=30, grid_line_width=1, occupied_px=(0,0,255), grid_line_px=(0,0,0)):
        self.unit_res = unit_res
        self.grid_line_px = grid_line_px
        self.occupied_px = occupied_px
        self.grid_line_width = grid_line_width

    def display_board(self, board_obj):
        unit_res = self.unit_res
        grid_line_width = self.grid_line_width 
        grid_line_px = self.grid_line_px
        occupied_px = self.occupied_px

        n = board_obj.n

        img_width = unit_res*n+n-grid_line_width 
        img_height = unit_res*n+n-grid_line_width  
        board_img = np.ones((img_height,img_width,3), dtype=np.uint8)
        board_img *= 255 # all to white

        # first, generate grid lines
        idx_x = 0
        idx_y = 0
        for x in xrange(n-1):
            idx_x += unit_res + grid_line_width
            board_img[:,idx_x-1] = grid_line_px
        for y in xrange(n-1):
            idx_y += unit_res + grid_line_width
            board_img[idx_y-1,:] = grid_line_px

        mr,mc = np.where(board_obj.pieces==1)
        for x,y in zip(mc,mr):
            self.fill_board_img_square(board_img, (x,y), occupied_px)
        return board_img

    def fill_board_img_square(self, board_img, square, fill_px):
        assert(type(board_img) == np.ndarray)
        assert(type(square) == tuple and type(fill_px) == tuple)
        
        r = self.unit_res
        gl_width = self.grid_line_width

        x,y = square
        start_x = x * r + gl_width * x
        start_y = y * r + gl_width * y
        board_img[start_y:start_y+r,start_x:start_x+r] = fill_px

    def fill_board_squares(self, board_obj, square_list, fill_px):
        board_img = self.display_board(board_obj)
        if type(square_list) != list:
            square_list = [square_list]
        for sq in square_list:
            self.fill_board_img_square(board_img, sq, fill_px)
        return board_img

if __name__ == '__main__':
    # CREATE VIEWER OF VALID MOVES AND NEXT STATE
    import cv2
    RED = (0,0,255)
    BLACK = (0,0,0)

    n = 6
    g = TetrisGame(n)
    b = Board(n)
    b_renderer = BoardRenderer(unit_res=30)

    rand_box_list = g.generateRandomBoxList()
    total_actions = g.getActionSize()

    print("Rand Box LIST:", rand_box_list)

    for box_size in rand_box_list:
        print()
        valid_actions = g.getValidMoves(b.pieces, [box_size])
        if valid_actions[-1] == 1:
            print("NO VALID ACTIONS FOR:", box_size)
            continue
        # pick random valid move
        valid_action_idx = np.where(valid_actions==1)[0]
        rand_action = random.choice(valid_action_idx)
        valid_action_squares = [g.boardIndexToSquare(g.action_list[action][0]) for action in valid_action_idx] 
        rand_action_square = g.boardIndexToSquare(g.action_list[rand_action][0])

        print("For box size:", box_size, "Valid squares:", valid_action_squares)
        print("Picking square:", rand_action_square)

        rand_action_x, rand_action_y = rand_action_square
        box_w, box_h = box_size
        proposed_board_img = b_renderer.fill_board_squares(b, [(rand_action_x+w,rand_action_y+h) for w in xrange(box_w) for h in xrange(box_h)], (0,255,0))
        cv2.imshow('proposed_board', proposed_board_img)

        b.pieces = g.getNextState(b.pieces, rand_action)

        board_img = b_renderer.display_board(b)
        cv2.imshow('board', board_img)

        print("Occupied cells: %d of %d, Score: %.3f"%(b.count_occupied(), b.total_cnt, b.get_score()))
        cv2.waitKey(0)

