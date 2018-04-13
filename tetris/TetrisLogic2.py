import numpy as np
import cv2
import random 

'''
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

class Board2(object):

    def __init__(self, r, c, n):
        "Set up initial board configuration."

        self.r = r 
        self.c = c
        self.n = n

        n_cells = n * 2  # x,y for each 

        assert n > 0 and c % 2 == 0

        # if c%2==1:
        #     self.rows = n / (c / 2) + 1
        # else:
        self.rows = n_cells / c + int(n_cells / c > 0)
        # self.rows = n_cells / (c - 1) + 1
        self.rows = r + self.rows 
        self.cols = c

        self.total_cells = self.rows * self.cols
        self.total_actions = self.r * self.c * self.n

        self.pieces = None

        self.reset()

    def reset(self):
        self.pieces = np.zeros((self.rows,self.cols), dtype=np.int8)
        box_list = self.generate_boxes(min_width=1, max_width=int(self.c/2) + 1, min_height=1, max_height=int(self.r/3))
        self._fill_pieces_with_box_list(box_list)
        self.box_list_area = self.calculate_box_list_area()

    def setBoard(self, board_pieces):
        self.pieces = board_pieces
        self.box_list_area = self.calculate_box_list_area()

    def _fill_pieces_with_box_list(self, box_list):
        r = 0
        c = 0
        for ix, box in enumerate(box_list):
            if ix >= self.n:
                break
            w,h = box

            if (self.c - c) <= 1:
                r = r + 1
                c = 0
            start_y = self.r+r
            if start_y >= self.rows:
                break

            self.pieces[start_y,c] = w
            self.pieces[start_y,c+1] = h
            c += 2

    def calculate_box_list_area(self):
        cache = self.pieces[self.r:].copy()
        if self.c % 2 != 0:
            cache = cache[:,:-1]
        cache = cache.flatten()
        box_list_area = sum([cache[i] * cache[i+1] for i in xrange(0,len(cache),2)])
        return int(box_list_area)

    def generate_boxes(self, min_width=1, max_width=4, min_height=1, max_height=2):
        boxes = []
        total_cells = self.r * self.c
        acc_cells = 0
        while acc_cells < total_cells and len(boxes) < self.n:
            w = random.randint(min_width, max_width)
            h = random.randint(min_height, max_height)
            acc_cells += w * h
            boxes.append((w,h))

        # then sort, smallest to biggest
        boxes = sorted(boxes, key=lambda bx: bx[0] * bx[1])
        # then sort same area boxes, from biggest width to smallest
        idx = 0
        total = len(boxes)
        sorted_boxes = []
        while idx < total:
            box1 = boxes[idx]
            cur_area = box1[0] * box1[1]
            same_area_boxes = [box1]
            for ix in xrange(idx + 1, total):
                box2 = boxes[ix]
                if box2[0] * box2[1] != cur_area:
                    break
                same_area_boxes.append(box2)
            sorted_boxes.extend(sorted(same_area_boxes, key=lambda bx: bx[0]))
            idx += len(same_area_boxes)
        return sorted_boxes

    def is_full(self):
        return np.all(self.pieces[:self.r]==1)

    def get_score(self):
        occ_cnt = self.get_occupied_count()
        half_cnt = min(self.box_list_area, self.r * self.c) / 2.
        occ_score = (float(occ_cnt - half_cnt) / half_cnt)# ** 2
        # occ_score = -occ_score if occ_cnt < half_cnt else occ_score
        return occ_score

    def get_occupied_count(self):
        return int(np.sum(self.pieces[:self.r]))  # since occupied are 1, non-occ are 0

    def is_valid_placement(self, square, box_size):
        x,y = square
        w,h = box_size

        assert w!=0 and h!=0
        assert x < self.c and y < self.r

        if self.pieces[y,x]==0: # not occupied
            if (x+w-1) < self.c and (y+h-1) < self.r: 
                if np.sum(self.pieces[y:y+h,x:x+w]) == 0:  # none of the placement cells are occupied
                    if (y+h) < self.r:  # if not on ground
                        # CHECK IF placement is on top of a sufficient number of occupied cells, relative to box width
                        return np.sum(self.pieces[y+h,x:x+w]) >= w # int(w/2)+1
                    return True
        return False

    def get_legal_moves(self, box_size):
        """Returns all the legal moves for the box size
        """
        # assert len(box_size) == 2  # box_size: w,h
        moves = set()  # stores the legal moves.
        (w,h) = box_size

        for y in xrange(self.r):
            for x in xrange(self.c):
                square = (x,y)
                if self.is_valid_placement(square, box_size):
                    moves.add(square)
        return list(moves)

    def get_legal_moves_all(self):
        legal_moves = []
        for box_idx in xrange(self.n):
            w,h = self.get_box_size_from_idx(box_idx)
            if w == 0:
                continue
            box_legal_moves = self.get_legal_moves((w,h))
            for mov in box_legal_moves:
                legal_moves.append(self.get_action_from_square_and_box_idx(mov, box_idx)) # convert to actions
        return legal_moves

    def has_legal_moves(self, box_size):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        assert len(box_size) == 2  # box_size: w,h
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in xrange(self.r):
            for x in xrange(self.c):
                square = (x,y)
                if self.is_valid_placement(square, box_size):
                    return True
        return False

    def has_legal_moves_all(self):
        for box_idx in xrange(self.n):
            w,h = self.get_box_size_from_idx(box_idx)
            if w > 0 and self.has_legal_moves((w,h)):
                return True
        return False

    def move_box(self, move, box_size):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        #Much like move generation, start at the new piece's square and
        #follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        x,y = move
        assert x < self.c and y < self.r
        w,h = box_size
        self.pieces[y:y+h,x:x+w] = 1

    def get_box_size_from_idx(self, box_idx):
        r = box_idx * 2 / self.c
        c = box_idx * 2 % self.c
        box_cells = self.pieces[self.r + r,c:c+2]
        w, h = box_cells
        return (w,h)

    def get_square_and_box_size_from_action(self, action):
        box_idx = action / (self.r * self.c)
        square_idx = action % (self.r * self.c)
        w,h = self.get_box_size_from_idx(box_idx)
        if w == 0:
            return None, None, box_idx
        x,y = self.boardIndexToSquare(square_idx)
        return (x,y), (w,h), box_idx

    def get_action_from_square_and_box_idx(self, square, box_idx):
        x, y = square
        return box_idx * self.r * self.c + y * self.c + x

    def is_action_valid(self, action):
        sq, box_size, box_idx = self.get_square_and_box_size_from_action(action)
        if sq is None:
            return False
        return self.is_valid_placement(sq, box_size)

    def execute_move(self, action):
        sq, box_size, box_idx = self.get_square_and_box_size_from_action(action)
        if sq is None:
            return
        x,y = sq
        w,h = box_size
        self.pieces[y:y+h,x:x+w] = 1

        # remove box idx
        cache_flat = self.pieces[self.r:].flatten()
        cache_flat = np.delete(cache_flat, [box_idx*2, box_idx*2+1])
        cache_flat = np.hstack((cache_flat,[0,0]))
        self.pieces[self.r:] = np.reshape(cache_flat, (len(cache_flat)/self.c, self.c))

    def boardIndexToSquare(self, idx):
        x = int(idx%self.c)
        y = int(idx/self.c) 
        return x,y

from TetrisLogic import BoardRenderer

class BoardRenderer2(BoardRenderer):

    def __init__(self, unit_res=30, grid_line_width=1, occupied_px=(0,0,255), box_px=(255,0,0), grid_line_px=(0,0,0), text_px=(0,0,0)):
        super(BoardRenderer2, self).__init__(unit_res, grid_line_width, occupied_px, box_px, grid_line_px, text_px)

    def display_board(self, board_obj):
        unit_res = self.unit_res
        grid_line_width = self.grid_line_width 
        grid_line_px = self.grid_line_px

        r = board_obj.r
        c = board_obj.c
        n = board_obj.n

        nr = board_obj.rows
        img_width = unit_res*c+c-grid_line_width 
        img_height = unit_res*nr+nr-grid_line_width  
        board_img = np.ones((img_height,img_width,3), dtype=np.uint8)
        board_img *= 255 # all to white

        # first, generate grid lines
        idx_x = 0
        idx_y = 0
        for x in xrange(c-1):
            idx_x += unit_res + grid_line_width
            board_img[:,idx_x-1] = grid_line_px
        for y in xrange(nr-1):
            idx_y += unit_res + grid_line_width
            board_img[idx_y-1,:] = grid_line_px

        mr,mc = np.where(board_obj.pieces[:r]==1)
        for x,y in zip(mc,mr):
            self.fill_board_img_square(board_img, (x,y), self.occupied_px)
        # for box_ix in xrange(n):
        #     box_size = board_obj.get_box_size_from_idx(box_ix)
        for y in xrange(r, nr):
            for x,cell in enumerate(board_obj.pieces[y]):
                if cell == 0:
                    continue
                self.fill_board_img_square(board_img, (x,y), self.box_px, "%d"%(cell))
        return board_img

if __name__ == '__main__':

    r = 6
    c = 8
    n = 9
    b = Board2(r,c,n)
    print(b.pieces)

    b_renderer = BoardRenderer2(unit_res=30)
    board_img = b_renderer.display_board(b)
    cv2.imshow('board', board_img)
    cv2.waitKey(0)

    # b.execute_move(30)
    # board_img = b_renderer.display_board(b)
    # cv2.imshow('board', board_img)
    # cv2.waitKey(0)


    while b.has_legal_moves_all():
        valid_actions = b.get_legal_moves_all()
        action = valid_actions[0]
        board_img = b_renderer.draw_action(b, action)
        cv2.imshow('board', board_img)
        b.execute_move(action)
        cv2.waitKey(0)        
