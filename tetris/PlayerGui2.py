# import the necessary packages
import cv2
from PlayerGui import PlayerGUI

class PlayerGUI2(PlayerGUI):
    def __init__(self, name, game, b_renderer=None):

        super(PlayerGUI2, self).__init__(name,game,b_renderer)

        self.end_key = "q"

    def setBoard(self, board_obj):
        g = self.game
        assert g.r == board_obj.r and g.c == board_obj.c and g.n == board_obj.n
        self.board = board_obj
        self.board_img = self.b_renderer.display_board(self.board)
        self.board_img_copy = self.board_img.copy()

    def _get_box_idx(self, square):
        x,y = square
        assert y >= self.board.r and x < self.board.c
        d_y = y - self.board.r
        box_idx = (d_y * self.board.c + x) / 2
        return box_idx

    def _is_in_cache(self, square):
        rt = square[1] >= self.board.r  # below n x n grid is cache
        return rt

if __name__ == '__main__':
    from TetrisLogic2 import Board2, BoardRenderer2
    from TetrisGame2 import TetrisGame2

    r = 6
    c = 8
    n = 9
    b = Board2(r,c,n)

    g = TetrisGame2(r,c,n)

    b_renderer = BoardRenderer2(unit_res=30)

    p_gui = PlayerGUI2("PlayerOne", g, b_renderer)
    p_gui.play(b)