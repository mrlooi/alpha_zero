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
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n, m):
        "Set up initial board configuration."

        self.n = n
        self.m = m
        self.rows = n + m
        self.cols = n

        self.total_cells = self.rows * self.cols
        self.total_actions = self.n * self.n * self.m

        self.pieces = None
        # self.box_list = []

        self.reset()

    def reset(self):
        self.pieces = np.zeros((self.n + self.m,self.n), dtype=np.int8)
        box_list = self.generate_boxes(min_width=1, max_width=int(self.n/2) + 1, min_height=1, max_height=int(self.n/3))
        self._fill_pieces_with_box_list(box_list)
        self.box_list_cells = self.calculate_box_list_area()

    def setBoard(self, board_pieces):
        self.pieces = board_pieces
        self.box_list_cells = self.calculate_box_list_area()

    def _fill_pieces_with_box_list(self, box_list):
        start_y = self.n
        for ix, box in enumerate(box_list):
            if ix >= self.m:
                break
            w,h = box
            self.pieces[start_y,:w] = h
            start_y += 1

    def calculate_box_list_area(self):
        return int(np.sum(self.pieces[self.n:]))

    def generate_boxes(self, min_width=1, max_width=4, min_height=1, max_height=2):
        boxes = []
        total_cells = self.n * self.n
        acc_cells = 0
        while acc_cells < total_cells and len(boxes) < self.m:
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
        return np.all(self.pieces[:self.n]==1)

    def get_score(self):
        occ_cnt = self.get_occupied_count()
        half_cnt = min(self.box_list_cells, self.n * self.n) / 2.
        occ_score = (float(occ_cnt - half_cnt) / half_cnt)# ** 2
        # occ_score = -occ_score if occ_cnt < half_cnt else occ_score
        return occ_score

    def get_occupied_count(self):
        return int(np.sum(self.pieces[:self.n]))  # since occupied are 1, non-occ are 0

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    # def countDiff(self, color):
    #     """Counts the extra # pieces of the given color
    #     (1 for white, -1 for black, 0 for empty spaces)"""
    #     count = 0
    #     for y in range(self.n):
    #         for x in range(self.n):
    #             if self[x][y]==color:
    #                 count += 1
    #             if self[x][y]==-color:
    #                 count -= 1
    #     return count


    def is_valid_placement(self, square, box_size):
        x,y = square
        w,h = box_size

        assert w!=0 and h!=0
        assert x < self.n and y < self.n

        if self.pieces[y,x]==0: # not occupied
            if (x+w-1) < self.n and (y+h-1) < self.n: 
                if np.sum(self[y:y+h,x:x+w]) == 0:  # none of the placement cells are occupied
                    if (y+h) < self.n:  # if not on ground
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

        for y in xrange(self.n):
            for x in xrange(self.n):
                square = (x,y)
                if self.is_valid_placement(square, box_size):
                    moves.add(square)
        return list(moves)

    def get_legal_moves_all(self):
        legal_moves = []
        for box_idx in xrange(self.m):
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
        for y in range(self.n):
            for x in range(self.n):
                square = (x,y)
                if self.is_valid_placement(square, box_size):
                    return True
        return False

    def has_legal_moves_all(self):
        for box_idx in xrange(self.m):
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
        assert(x < self.n and y < self.n)
        w,h = box_size
        self.pieces[y:y+h,x:x+w] = 1

    def get_box_size_from_idx(self, box_idx):
        box_cells = self.pieces[self.n + box_idx]
        w = int(np.sum(box_cells > 0))
        if w == 0:
            return (0,0)
        h = int(box_cells[0])  # assumes first index always occupied
        return (w,h)

    def get_square_and_box_size_from_action(self, action):
        box_idx = action / (self.n * self.n)
        square_idx = action % (self.n * self.n)
        w,h = self.get_box_size_from_idx(box_idx)
        if w == 0:
            return None, None, box_idx
        x,y = self.boardIndexToSquare(square_idx)
        return (x,y), (w,h), box_idx

    def get_action_from_square_and_box_idx(self, square, box_idx):
        x, y = square
        return box_idx * self.n * self.n + y * self.n + x

    def execute_move(self, action):
        sq, box_size, box_idx = self.get_square_and_box_size_from_action(action)
        if sq is None:
            return
        x,y = sq
        w,h = box_size
        self.pieces[y:y+h,x:x+w] = 1
        # self.pieces[self.n+box_idx] = 0 
        self.pieces[self.n+box_idx:-1] = self.pieces[self.n+box_idx+1:]
        self.pieces[-1] = 0  # move up

    def boardIndexToSquare(self, idx):
        x = int(idx%self.n)
        y = int(idx/self.n) 
        return x,y

    def _discover_move(self, origin, direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        #initialize variables
        flips = [origin]

        for x, y in Board._increment_move(origin, direction, self.n):
            #print(x,y)
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                #print(flips)
                return flips

        return []

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)): 
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])


class BoardRenderer(object):

    def __init__(self, unit_res=30, grid_line_width=1, occupied_px=(0,0,255), box_px=(255,0,0), grid_line_px=(0,0,0), text_px=(0,0,0)):
        self.unit_res = unit_res
        self.grid_line_px = grid_line_px
        self.occupied_px = occupied_px
        self.grid_line_width = grid_line_width
        self.box_px = box_px
        self.text_px = text_px

    def display_board(self, board_obj):
        unit_res = self.unit_res
        grid_line_width = self.grid_line_width 
        grid_line_px = self.grid_line_px

        n = board_obj.n
        m = board_obj.m

        nr = n + m # total rows
        img_width = unit_res*n+n-grid_line_width 
        img_height = unit_res*nr+nr-grid_line_width  
        board_img = np.ones((img_height,img_width,3), dtype=np.uint8)
        board_img *= 255 # all to white

        # first, generate grid lines
        idx_x = 0
        idx_y = 0
        for x in xrange(n-1):
            idx_x += unit_res + grid_line_width
            board_img[:,idx_x-1] = grid_line_px
        for y in xrange(nr-1):
            idx_y += unit_res + grid_line_width
            board_img[idx_y-1,:] = grid_line_px

        mr,mc = np.where(board_obj.pieces[:n]==1)
        for x,y in zip(mc,mr):
            self.fill_board_img_square(board_img, (x,y), self.occupied_px)
        for ix in xrange(m):
            y = n+ix
            for x,cell in enumerate(board_obj.pieces[y]):
                if cell == 0:
                    continue
                self.fill_board_img_square(board_img, (x,y), self.box_px, "%d"%(cell))
        return board_img

    def fill_board_img_square(self, board_img, square, fill_px, text=None):
        assert(type(board_img) == np.ndarray)
        assert(type(square) == tuple and type(fill_px) == tuple)
        
        r = self.unit_res
        gl_width = self.grid_line_width

        x,y = square
        start_x = x * r + gl_width * x
        start_y = y * r + gl_width * y
        board_img[start_y:start_y+r,start_x:start_x+r] = fill_px

        if text:
            pos = (start_x + r / 2, start_y + r / 2)
            font_scale = r / 50.
            cv2.putText(board_img, text, pos, cv2.FONT_HERSHEY_COMPLEX, font_scale, self.text_px)

    def fill_board_squares(self, board_obj, square_list, fill_px):
        board_img = self.display_board(board_obj)
        if type(square_list) != list:
            square_list = [square_list]
        for sq in square_list:
            self.fill_board_img_square(board_img, sq, fill_px)
        return board_img

    def draw_action(self, board_obj, action, action_px=(0,255,0)):
        b = board_obj 
        sq, box_sz, _ = b.get_square_and_box_size_from_action(action)
        box_w, box_h = box_sz

        board_img = self.fill_board_squares(b, [(sq[0]+w,sq[1]+h) for w in xrange(box_w) for h in xrange(box_h)], action_px)
        return board_img

if __name__ == '__main__':

    n = 6
    m = 10
    b = Board(n, m)

    box_sizes = [(3,2),(1,1),(2,2),(3,2),(3,1),(3,1),(5,2),(5,2),(2,2)] # (w,h), ...
    moves =     [(0,0),(5,5),(5,4),(0,4),(3,4),(1,3),(2,1),(1,1),(1,2)] # (x,y), ...
    for ix,move in enumerate(moves):
        box_size = box_sizes[ix]
        if b.is_valid_placement(move, box_size):
            print("Move valid!", move, "Box size:", box_size)
            b.move_box(move, box_size)
            print(b.pieces[:n])

    print(b.get_legal_moves((2,1)))

    b.reset()

    b_renderer = BoardRenderer(unit_res=30)
    board_img = b_renderer.display_board(b)
    cv2.imshow('board', board_img)
    cv2.waitKey(0)

    b.execute_move(30)
    board_img = b_renderer.display_board(b)
    cv2.imshow('board', board_img)
    cv2.waitKey(0)

    for action in sorted(b.get_legal_moves_all()):
        board_img = b_renderer.draw_action(b, action)
        cv2.imshow('board', board_img)
        cv2.waitKey(0)        
