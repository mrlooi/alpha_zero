from __future__ import print_function
import sys
sys.path.append('..')
# from Game import Game
# from .TetrisLogic import Board, BoardRenderer
from TetrisLogic2 import Board2, BoardRenderer2
import numpy as np
import random


class TetrisGame2(object):
    def __init__(self, r,c,n):
        self.r = r
        self.c = c
        self.n = n

        self.board = Board2(r, c, n)

    def getInitBoard(self):
        # return initial board (numpy board)
        self.board.reset()
        return self.board

    def getBoardSize(self):
        # (a,b) tuple
        return (self.board.cols, self.board.rows)

    def getActionSize(self):
        # return number of actions
        # return self.n*self.n + 1
        # box_list_cnt = len(getInitBoxList())
        # board_sz = np.size(getInitBoard())
        # return board_sz * box_list_cnt + 1
        
        return self.board.total_actions + 1  # + 1 for end game action

    def boardIndexToSquare(self, idx):
        return self.board.boardIndexToSquare(idx)

    def getNextState(self, board_obj, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.getActionSize():
            return board_obj
        b = board_obj
        b.execute_move(action)
        return b

    def getValidMoves(self, board_obj):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = board_obj

        legalMoves = b.get_legal_moves_all()

        if len(legalMoves)==0:
            valids[-1]=1
        else:
            for ix in legalMoves:
                valids[ix]=1
        return np.array(valids)

    def getGameEnded(self, board_obj):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        if board_obj.has_legal_moves_all():
            return False
        return True

    def getScore(self, board_obj):
        return board_obj.get_score()

    def getCanonicalForm(self, board_obj):
        # return state if player==1, else return -state if player==-1
        return board_obj

    def getSymmetries(self, board_obj, pi):
        '''
        pi is the policy output (size: total actions)
        '''
        # horizontal flip only 
        assert(len(pi) == self.getActionSize())  # 1 for pass
        l = []

        # for i in range(1, 5):
        board = board_obj.pieces
        for j in [True, False]:
            newB = board.copy() # np.rot90(board, i)
            if j:
                newB[:self.r] = np.fliplr(newB[:self.r])  # only flip the top part!
                newPi = self.flip_pi_LR(board_obj, pi)
                newPi = list(newPi) + [pi[-1]]
            else:
                newPi = list(pi)
            l += [(newB, newPi)]
        return l

    def flip_pi_LR(self, board_obj, pi):
        n = board_obj.n
        r = board_obj.r
        c = board_obj.c
        total_actions = board_obj.total_actions
        assert len(pi) >= total_actions == r * c * n
        newPi_m = np.reshape(pi.copy()[:total_actions], (n, r, c))

        for box_ix, mask in enumerate(newPi_m):
            # newPi = np.array([np.fliplr(pi_) for pi_ in newPi])  # CHECK IT!
            box_w,_ = board_obj.get_box_size_from_idx(box_ix)
            if box_w == 0:
                continue
            mask = np.fliplr(mask)
            shift_x = box_w - 1  # shift to the left
            mask[:,:c-shift_x] = mask[:,shift_x:]
            mask[:,c-shift_x:] = 0
            newPi_m[box_ix] = mask
        return newPi_m.ravel()

    def stringRepresentation(self, board_obj):
        # 8x8 numpy array (canonical board)
        return board_obj.pieces.tostring()


if __name__ == '__main__':
    # CREATE VIEWER OF VALID MOVES AND NEXT STATE
    import cv2
    import copy 

    RED = (0,0,255)
    BLACK = (0,0,0)

    n = 10
    r = 6 
    c = 8
    g = TetrisGame2(r, c, n)
    b_renderer = BoardRenderer2(unit_res=30)

    b = Board2(r, c, n)
    flipped_b = copy.deepcopy(b)
    flipped_b.pieces[:r] = np.fliplr(flipped_b.pieces[:r])

    total_actions = g.getActionSize()

    board_img = b_renderer.display_board(b)
    flipped_board_img = b_renderer.display_board(flipped_b)
    cv2.imshow('board', board_img)
    cv2.imshow('flipped_board', flipped_board_img)
    cv2.waitKey(0)

    while not g.getGameEnded(b):
        valid_actions = g.getValidMoves(b)
        if valid_actions[-1] == 1:
            print("NO MORE VALID ACTIONS")
            break

        valid_action_idx = np.where(valid_actions==1)[0]
        rand_action = random.choice(valid_action_idx)

        # sq, box_sz, _ = b.get_square_and_box_size_from_action(rand_action)
        # box_w, box_h = box_sz
        # print("Picked random action %d -> box dims: (%d, %d), Square: (%d, %d)"%(rand_action, box_w, box_h, sq[0], sq[1]))

        rand_action_onehot = np.zeros(len(valid_actions))
        rand_action_onehot[rand_action] = 1
        rand_action_onehot_flipped = g.flip_pi_LR(flipped_b, rand_action_onehot)
        rand_action_flipped = np.where(rand_action_onehot_flipped==1)[0][0]

        proposed_board_img = b_renderer.draw_action(b, rand_action)
        proposed_flipped_board_img = b_renderer.draw_action(flipped_b, rand_action_flipped)

        cv2.imshow('proposed_board', proposed_board_img)
        cv2.imshow('flipped_proposed_board', proposed_flipped_board_img)

        b = g.getNextState(b, rand_action)
        flipped_b = g.getNextState(flipped_b, rand_action_flipped)

        board_img = b_renderer.display_board(b)
        flipped_board_img = b_renderer.display_board(flipped_b)
        cv2.imshow('board', board_img)
        cv2.imshow('flipped_board', flipped_board_img)

        print("Occupied cells: %d of available %d, Score: %.3f"%(b.get_occupied_count(), min(b.box_list_area, n * n), b.get_score()))
        cv2.waitKey(0)

    # b = Board(n, m)
    # legal_moves = sorted(b.get_legal_moves_all())
    # valid_actions = np.zeros(b.total_actions)
    # valid_actions[legal_moves] = 1
    # valid_actions_flipped = g.flip_pi_LR(b, valid_actions)
    # valid_actions_flipped_idx = np.where(valid_actions_flipped==1)[0]

    # for action in valid_actions_flipped_idx:
    #     sq, box_sz, _ = b.get_square_and_box_size_from_action(action)

    #     box_w, box_h = box_sz

    #     board_img = b_renderer.fill_board_squares(b, [(sq[0]+w,sq[1]+h) for w in xrange(box_w) for h in xrange(box_h)], (0,255,0))
    #     cv2.imshow('nboard', board_img)
    #     cv2.waitKey(0)        
