import numpy as np
import random 


class BoxListGenerator(object):
    def __init__(self, min_x=1, max_x=1, min_y=1, max_y=1, min_z=1, max_z=1):
        assert max_x >= min_x > 0 and max_y >= min_y > 0 and max_z >= min_z > 0

        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    def generate(self, n, max_cells=None, sort=True):
        boxes = []
        acc_cells = 0
        while len(boxes) < n:
            if max_cells is not None and acc_cells >= max_cells:
                break
            w = random.randint(self.min_x, self.max_x)
            h = random.randint(self.min_y, self.max_y)
            d = random.randint(self.min_z, self.max_z) # depth (z)
            acc_cells += w * h * d
            boxes.append((w,h,d))

        if sort:
            boxes = self.sort_box_list(boxes)
        return boxes

    def sort_box_list(self, box_list):
        str_boxes = sorted(["%d_%d_%d"%(i[0],i[1],i[2]) for i in box_list])
        sorted_boxes = [np.array(b.split("_")).astype(np.int32) for b in str_boxes]
        return sorted_boxes

class Board(object):

    def __init__(self, x, y, z, n, box_list_gen=None):
        "Set up initial board configuration."

        self.x = x 
        self.y = y
        self.z = z
        self.n = n

        n_cells = n * 3  # x,y,z for each 

        assert n > 0 and x % 3 == 0

        y_rows = ((n_cells / x) + int(n_cells % x > 0))
        self.cache_rows = y_rows / y + int(y_rows % y > 0)

        self.len_x = x
        self.len_y = y
        self.len_z = z + self.cache_rows

        self.total_cells = self.len_x * self.len_y * self.len_z
        self.total_actions = self.x * self.y * self.z * self.n

        self.box_list_generator = BoxListGenerator(min_x=1, max_x=int(self.len_x/2) + 1, min_y=1, max_y=int(self.y/2)+1, min_z=1, max_z=2)
        if box_list_gen is not None:
            assert type(box_list_gen) == BoxListGenerator
            self.box_list_generator = box_list_gen

        self.pieces = None

        self.reset()

    def reset(self):
        self.pieces = np.zeros((self.len_z,self.len_y,self.len_x), dtype=np.int8)
        box_list = self.generate_boxes()
        self._fill_pieces_with_box_list(box_list)
        self.box_list_area = self.calculate_box_list_area()

    def setBoard(self, board_pieces):
        self.pieces = board_pieces
        self.box_list_area = self.calculate_box_list_area()

    def _fill_pieces_with_box_list(self, box_list):
        data_flat = np.zeros(self.cache_rows * self.y * self.x)
        data_flat[:len(box_list) * 3] = np.array(box_list).flatten()
        data = data_flat.reshape((self.cache_rows, self.y, self.x))
        # print(data_flat.shape)
        self.pieces[self.z:] = data
        # print(data)

    def calculate_box_list_area(self):
        cache = self.pieces[self.z:,:].copy()
        if self.x % 3 != 0:
            cache = cache[:,:,:-(self.x % 3)]
        cache = cache.flatten()
        box_list_area = sum([cache[i] * cache[i+1] * cache[i+2] for i in xrange(0,len(cache),3)])
        return int(box_list_area)

    def generate_boxes(self):
        sorted_boxes = self.box_list_generator.generate(n=self.n, max_cells=self.x * self.y * self.z, sort=True)

        # print(sorted_boxes)
        # print(len(sorted_boxes))
        return sorted_boxes

    def is_full(self):
        return np.all(self.pieces[:self.z]==1)

    def get_occupied_count(self):
        return int(np.sum(self.pieces[:self.z]))  # since occupied are 1, non-occ are 0

    def get_score(self):
        occ_cnt = self.get_occupied_count()
        half_cnt = min(self.box_list_area, self.z * self.y * self.x) / 2.
        occ_score = (float(occ_cnt - half_cnt) / half_cnt)# ** 2
        # occ_score = -occ_score if occ_cnt < half_cnt else occ_score
        return occ_score

    def is_valid_placement(self, square, box_size):
        x,y,z = square
        w,h,d = box_size

        assert w!=0 and h!=0 and d!=0
        assert x < self.x and y < self.y and z < self.z

        if self.pieces[z,y,x]==0: # not occupied
            if (x+w-1) < self.x and (y+h-1) < self.y and (z+d-1) < self.z: 
                if np.sum(self.pieces[z:z+d,y:y+h,x:x+w]) == 0:  # none of the placement cells are occupied
                    if (z+d) < self.z:  # if not on ground
                        # CHECK IF placement is on top of a sufficient number of occupied cells
                        is_stable = np.sum(self.pieces[z+d,y:y+h,x:x+w]) >= w*h
                        if not is_stable:
                            return False
                    if y > 0:  # if not next to grid wall
                        is_at_wall = np.sum(self.pieces[z:z+d,y-1,x:x+w]) >= (w*d / 2 + 1)  # at least bigger than half next to it
                        return is_at_wall 
                    return True
        return False

    def get_legal_squares(self, box_size):
        """Returns all the legal moves for the box size
        """
        # assert len(box_size) == 2  # box_size: w,h
        moves = set()  # stores the legal moves.
        w,h,d = box_size

        for z in xrange(self.z):
            for y in xrange(self.y):
                for x in xrange(self.x):
                    square = (x,y,z)
                    if self.is_valid_placement(square, box_size):
                        moves.add(square)
        return list(moves)        

    def get_legal_moves(self, box_idx):
        """Returns all the legal moves for the box size
        """
        # assert len(box_size) == 2  # box_size: w,h
        box_size = self.get_box_size_from_idx(box_idx)
        if box_size[0] == 0:
            return []
        legal_squares = self.get_legal_squares(box_size)
        # print(sorted(legal_squares))
        return [self.get_action_from_square_and_box_idx(sq, box_idx) for sq in legal_squares]

    def get_legal_moves_all(self):
        legal_moves = []
        for box_idx in xrange(self.n):
            legal_moves += self.get_legal_moves(box_idx)
        return legal_moves

    def has_legal_moves(self, box_size):
        assert len(box_size) == 3  # box_size: w,h,d

        legal_moves = self.get_legal_squares(box_size)
        return len(legal_moves) != 0

    def has_legal_moves_all(self):
        for box_idx in xrange(self.n):
            w,h,d = self.get_box_size_from_idx(box_idx)
            if w > 0 and self.has_legal_moves((w,h,d)):
                return True
        return False

    def get_box_size_from_idx(self, box_idx):
        assert box_idx < self.n

        x = box_idx * 3 % self.x
        y = box_idx * 3 / self.x
        z = y / self.y
        y = y - z * self.y
        box_cells = self.pieces[self.z + z, y, x:x+3]
        w, h, d = box_cells
        return (w,h,d)

    def get_action_from_square_and_box_idx(self, square, box_idx):
        x, y, z = square
        return box_idx * self.x * self.y * self.z + z * self.x * self.y + y * self.x + x

    def get_square_and_box_size_from_action(self, action):

        box_idx = action / (self.x * self.y * self.z)
        square_idx = action % (self.x * self.y * self.z)
        w,h,d = self.get_box_size_from_idx(box_idx)
        if w == 0:
            return None, None, box_idx
        x,y,z = self.boardIndexToSquare(square_idx)
        return (x,y,z), (w,h,d), box_idx

    def is_action_valid(self, action):
        sq, box_size, box_idx = self.get_square_and_box_size_from_action(action)
        if sq is None:
            return False
        return self.is_valid_placement(sq, box_size)

    def boardIndexToSquare(self, idx):
        z = idx / (self.x * self.y)
        rem = idx % (self.x * self.y)
        y = rem / self.x
        x = rem % self.x
        return x,y,z

    def move_box(self, square, box_size):
        x,y,z = square
        assert x < self.x and y < self.y and z < self.z
        w,h,d = box_size
        self.pieces[z:z+d,y:y+h,x:x+w] = 1

    def execute_move(self, action):
        sq, box_size, box_idx = self.get_square_and_box_size_from_action(action)
        if sq is None:
            return
        self.move_box(sq, box_size)

        # remove box idx
        cache_flat = self.pieces[self.z:].flatten()
        cache_flat = np.delete(cache_flat, [box_idx*3, box_idx*3+1,box_idx*3+2])
        cache_flat = np.hstack((cache_flat,[0,0,0]))
        self.pieces[self.z:] = np.reshape(cache_flat, (len(cache_flat) / (self.y * self.x), self.y, self.x))


if __name__ == '__main__':
    # from VoxelRender import BoardRenderer

    x = 6
    y = 7
    z = 3
    n = 10
    b = Board(x,y,z,n)

    random_box_idx = n / 3
    valid_actions = sorted(b.get_legal_moves(random_box_idx))
    cnt = 0
    for action in valid_actions:
        sq, box_sz, box_idx = b.get_square_and_box_size_from_action(action)
        print(sq, box_sz)

        cnt += 1
        # if cnt >= 5:
        #     break

    random_action = random.choice(valid_actions)
    print("Picking action:", random_action)

    # execute action 
    b.execute_move(random_action)

    print("Score %.3f"%(b.get_score()))