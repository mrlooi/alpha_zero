import numpy as np
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

    def __init__(self, n):
        "Set up initial board configuration."

        self.n = n
        self.total_cnt = n * n
        self.reset()

        # self.pieces = np.zeros((n,n))
        # self.boxes = self.generate_boxes()

    def reset(self):
        self.pieces = np.zeros((self.n,self.n))



    def generate_boxes(self, min_width=1, max_width=4, min_height=1, max_height=2):
        boxes = []
        total_cells = self.n * self.n
        acc_cells = 0
        while acc_cells < total_cells:
            w = random.randint(min_width, max_width)
            h = random.randint(min_height, max_height)
            acc_cells += w * h
            boxes.append((w,h))
        return boxes

    def remove_box(self, box_size):
        self.boxes.remove(box_size)

    def is_full(self):
        return np.all(self.pieces==1)

    def get_score(self):
        occ_cnt = self.count_occupied()
        half_cnt = self.total_cnt / 2
        occ_score = (float(occ_cnt - half_cnt) / half_cnt) ** 2
        occ_score = -occ_score if occ_cnt < half_cnt else occ_score
        return occ_score

    def count_occupied(self):
        return np.sum(self.pieces)  # since occupied are 1, non-occ are 0

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

    def execute_move(self, move, box_size):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        #Much like move generation, start at the new piece's square and
        #follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        x,y = move
        w,h = box_size
        self.pieces[y:y+h,x:x+w] = 1

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


if __name__ == '__main__':
    b = Board(6)

    box_sizes = [(3,2), (1,1),(2,2),(3,2),(3,1),(3,1),(5,2),(5,2)]
    moves = [(0,0),(5,5),(5,4),(0,4),(3,4),(1,3),(2,1),(1,1)] # (x,y), ...
    for ix,move in enumerate(moves):
        box_size = box_sizes[ix]
        if b.is_valid_placement(move, box_size):
            print("Move valid!", move)
            b.execute_move(move, box_size)
            print(b.pieces)
    print(b.get_legal_moves((2,1)))
