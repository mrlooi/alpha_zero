from __future__ import print_function
import sys
sys.path.append('..')
# from Game import Game
# from .TetrisLogic import Board, BoardRenderer
from VoxelLogic import Board
import numpy as np
import random

class VoxelGame(object):
    def __init__(self,x,y,z,n):
        self.x = x
        self.y = y
        self.z = z
        self.n = n

        self.board = Board(x,y,z,n)

    def getInitBoard(self):
        # return initial board (numpy board)
        self.board.reset()
        return self.board

    def getBoardSize(self):
        return (self.board.len_z, self.board.len_y, self.board.len_x)

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
        return not board_obj.has_legal_moves_all()

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
        b = board_obj
        board = board_obj.pieces
        for j in [True, False]:
            newB = board.copy() # np.rot90(board, i)
            if j:
                newB[:b.z] = np.array([np.fliplr(x) for x in b.pieces[:b.z]])  # only flip the top part!
                newPi = self.flip_pi_LR(board_obj, pi)
                newPi = list(newPi) + [pi[-1]]
            else:
                newPi = list(pi)
            l += [(newB, newPi)]
        return l

    def flip_pi_LR(self, board_obj, pi):
        b = board_obj
        n = b.n
        x = b.x
        y = b.y
        z = b.z

        total_actions = b.total_actions

        assert len(pi) >= total_actions == x * y * z * n
        newPi_m = np.reshape(np.array(pi)[:total_actions], (n, z, y, x))

        for box_ix, mask in enumerate(newPi_m):
            # newPi = np.array([np.fliplr(pi_) for pi_ in newPi])  # CHECK IT!
            box_w,_,_ = b.get_box_size_from_idx(box_ix)
            if box_w == 0:
                continue
            mask = np.array([np.fliplr(m) for m in mask])
            shift_x = box_w - 1  # shift to the left
            mask[:,:,:x-shift_x] = mask[:,:,shift_x:]
            mask[:,:,x-shift_x:] = 0
            newPi_m[box_ix] = mask
        return newPi_m.ravel()

    def stringRepresentation(self, board_obj):
        # 8x8 numpy array (canonical board)
        return board_obj.pieces.tostring()


if __name__ == '__main__':
    import copy

    n = 5
    x = 6 
    y = 4
    z = 3

    b = Board(x, y, z, n)
    g = VoxelGame(x, y, z, n)

    from VoxelRender import BoardRenderer
    b_renderer = BoardRenderer(name='Normal')
    b_renderer2 = BoardRenderer(name='Flipped')

    flipped_b = copy.deepcopy(b)

    while not g.getGameEnded(b):
        valid_actions = g.getValidMoves(b)
        if valid_actions[-1] == 1:
            print("NO MORE VALID ACTIONS")
            break

        valid_action_idx = np.where(valid_actions==1)[0]
        rand_action = random.choice(valid_action_idx)

        rand_action_onehot = np.zeros(len(valid_actions))
        rand_action_onehot[rand_action] = 1

        rand_action_onehot_flipped = g.flip_pi_LR(flipped_b, rand_action_onehot)
        rand_action_flipped = np.where(rand_action_onehot_flipped==1)[0][0]

        b_renderer.draw_action(b, rand_action)
        b_renderer2.draw_action(flipped_b, rand_action_flipped)

        b_renderer.show(1)

        b = g.getNextState(b, rand_action)
        flipped_b = g.getNextState(flipped_b, rand_action_flipped)

        b_renderer.display_board(b)
        b_renderer2.display_board(flipped_b)

        b_renderer.show(1)


        print("Occupied cells: %d of available %d, Score: %.3f"%(b.get_occupied_count(), min(b.box_list_area, x * y * z), b.get_score()))