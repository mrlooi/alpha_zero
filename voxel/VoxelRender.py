import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class BoardRenderer(object):
    def __init__(self, name = 'RenderOne', occupied_col = 'blue', text_col = 'black'):
        
        self.fig = plt.figure()
        self.fig.canvas.set_window_title(name)
        self.ax = self.fig.gca(projection='3d')

        self.occupied_col = occupied_col
        self.text_col = text_col


    def display_board(self, board):

        ax = self.fig.gca(projection='3d')
        ax.clear()

        # swap x-axis with z-axis for visualization
        voxels_reshape = np.swapaxes(board.pieces, 0, 2)
        # voxels_reshape = board.pieces

        occupied = np.zeros(voxels_reshape.shape, dtype=np.bool)
        occupied[voxels_reshape>0] = True
        occupied[:,:,board.z:] = False
        # set the colors of each object
        colors = np.empty(voxels_reshape.shape, dtype=object)
        # colors[link] = 'red'
        colors[occupied] = self.occupied_col
        # colors[cube2] = 'green'

        ax.voxels(colors, facecolors=colors, edgecolor='k')

        for z in xrange(board.z, board.len_z):
            for y in xrange(board.len_y):
                for x in xrange(board.len_x):
                    # v = voxels_reshape[z,y,x]
                    # if v > 0:
                    #     ax.text(z+.5, y+.5, x+0.5, "%d"%(v), color='red')
                    v = voxels_reshape[x,y,z]
                    if v > 0:
                        ax.text(x+.5, y+.5, z+0.5, "%d"%(v), color=self.text_col)

        ax.set_xlim3d(0, board.len_x)
        ax.set_ylim3d(0, board.len_y)
        ax.set_zlim3d(0, board.len_z)
        ax.view_init(220, 240)
        # for angle in range(180, 360):
        #     ax.view_init(210, angle)
        #     plt.draw()
        #     plt.pause(.001)

        return ax, voxels_reshape
        # plt.draw()
        # plt.pause(0)

    def fill_board_squares(self, board_obj, square_list, fill_col = 'green'):
        ax, voxels = self.display_board(board_obj)
        if type(square_list) != list:
            square_list = [square_list]

        colors = np.empty(voxels.shape, dtype=object)
        for sq in square_list:
            w,h,d = sq
            colors[w,h,d] = fill_col
        ax.voxels(colors, facecolors=colors, edgecolor='k')
        return ax

    def draw_box_from_square(self, board_obj, square, box_size, fill_col = 'green'):
        box_w, box_h, box_d = box_size
        sq = square
        fill_squares = [(sq[0]+w,sq[1]+h,sq[2]+d) for w in xrange(box_w) for h in xrange(box_h) for d in xrange(box_d)]
        board_img = self.fill_board_squares(board_obj, fill_squares, fill_col)
        return board_img

    def draw_action(self, board_obj, action, fill_col='green'):
        b = board_obj 
        sq, box_sz, _ = b.get_square_and_box_size_from_action(action)
        return self.draw_box_from_square(b, sq, box_sz, fill_col)

    def show(self, wait=1):
        if wait == 0:
            plt.show()
        else:
            plt.draw()
            plt.pause(wait)

if __name__ == '__main__':
    from VoxelLogic import Board 
    import random

    x = 9
    y = 3
    z = 6
    n = 20
    b = Board(x,y,z,n)

    renderer = BoardRenderer()
    ax = renderer.display_board(b)

    random_box_idx = n / 3
    valid_actions = sorted(b.get_legal_moves(random_box_idx))
    cnt = 0
    for action in valid_actions:
        sq, box_sz, box_idx = b.get_square_and_box_size_from_action(action)
        ax = renderer.draw_box_from_square(b, sq, box_sz, 'green')
        renderer.show(0.1)
    
        cnt += 1

    def random_run():
        while b.has_legal_moves_all():
            valid_actions = sorted(b.get_legal_moves_all())
            for action in valid_actions:
                ax = renderer.draw_action(b, action, 'green')
                renderer.show(0.001)

            random_action = random.choice(valid_actions)
            print("Picking action:", random_action)
            b.execute_move(random_action)

        ax, _ = renderer.display_board(b)

        renderer.show(0.1)
    random_run()

    print("Score %.3f"%(b.get_score()))