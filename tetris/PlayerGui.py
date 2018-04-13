# import the necessary packages
import cv2
 

class PlayerGUI(object):
    def __init__(self, name, game, b_renderer=None):
        self.refPt = []
        self.name = name
        self.cur_box = None

        self.game = game
        self.board = None
        self.b_renderer = BoardRenderer(unit_res=30) if b_renderer is None else b_renderer

        self.end_key = "q"

    def setBoard(self, board_obj):
        g = self.game
        assert g.n == board_obj.n and g.m == board_obj.m
        self.board = board_obj
        self.board_img = self.b_renderer.display_board(self.board)
        self.board_img_copy = self.board_img.copy()

    def play(self, board):
        self.setBoard(board)
        b = self.board

        if b is None:
            print("Please set the board object")
            return 

        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.click_and_play)
         
        print("Press '%s' button to quit game."%(self.end_key))

        score = -1
        while True:
            image = self.board_img_copy

            # display the image and wait for a keypress
            cv2.imshow(self.name, image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(self.end_key):
                score = self.game.getScore(self.board)
                print("Ended Game. Score: %.3f"%(score))
                break
         
        # close all open windows
        cv2.destroyAllWindows()

        return score

    def click_and_play(self, event, x, y, flags, param):
        image = self.board_img_copy 
        square = self.b_renderer.get_square_from_pixel_pos(self.board, (x,y))

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._is_in_cache(square): # picking from cache
                box_ix = self._get_box_idx(square)
                if box_ix == self.cur_box:  # deselect
                    self.cur_box = None
                    print("Deselected Box %d"%(box_ix+1))
                else:
                    box_sz = self.board.get_box_size_from_idx(box_ix)
                    if box_sz[0] == 0:
                        print("Please pick a valid box!")
                        return
                    self.cur_box = box_ix
                    print("Picked Box %d) W %d H %d"%(box_ix+1, box_sz[0], box_sz[1]))
            else:
                if self.cur_box is not None:  # has a box, and placing it in grid 
                    action = self.board.get_action_from_square_and_box_idx(square, self.cur_box)
                    if self.board.is_action_valid(action):
                        new_board = self.game.getNextState(self.board, action)
                        self.setBoard(new_board)
                        self.cur_box = None
                    else:
                        print("INVALID PLACEMENT")
                else:
                    print("PLEASE PICK A BOX FROM THE CACHE")

        elif event == cv2.EVENT_MOUSEMOVE:
            self.board_img_copy = self.board_img.copy()
            if not self._is_in_cache(square): # moving mouse to cache
                if self.cur_box is not None:
                    box_sz = self.board.get_box_size_from_idx(self.cur_box)
                    self.board_img_copy = self.b_renderer.draw_box_from_square(self.board, square, box_sz, (0,255,0))

    def _get_box_idx(self, square):
        return square[1] - self.board.n

    def _is_in_cache(self, square):
        rt = square[1] >= self.game.n  # below n x n grid is cache
        return rt

if __name__ == '__main__':
    from TetrisLogic import Board, BoardRenderer
    from TetrisGame import TetrisGame

    n = 12
    m = 15
    g = TetrisGame(n, m)
    b = Board(n, m)
    b_renderer = BoardRenderer(unit_res=30)

    p_gui = PlayerGUI("PlayerOne", g, b_renderer)
    p_gui.play(b)