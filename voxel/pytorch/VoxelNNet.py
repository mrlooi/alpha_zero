
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torch.autograd import Variable


class VoxelNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y, self.board_z = game.getBoardSize() 
        self.input_shape = (1, self.board_x, self.board_y, self.board_z)
        self.action_size = game.getActionSize()  
        self.args = args

        super(VoxelNNet, self).__init__()

        self.features = self._make_feature_layers()

        in_channels = self._get_conv_output(self.input_shape)

        fc1_channels = 512
        fc2_channels = 512
        self.fc1 = nn.Linear(in_channels, fc1_channels)
        self.fc_bn1 = nn.BatchNorm1d(fc1_channels)

        self.fc2 = nn.Linear(fc1_channels, fc2_channels)
        self.fc_bn2 = nn.BatchNorm1d(fc2_channels)

        self.fc3 = nn.Linear(fc2_channels, self.action_size)

        self.fc4 = nn.Linear(fc2_channels, 1)


    def _make_feature_layers(self):
        layers = []

        conv1_channels = 32
        conv2_channels = 64
        conv3_channels = 128
        conv4_channels = 256

        in_channels = 1 

        conv1 = [nn.Conv3d(in_channels, out_channels=conv1_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(conv1_channels), nn.ReLU(inplace=True)]
        conv2 = [nn.Conv3d(conv1_channels, out_channels=conv2_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(conv2_channels), nn.ReLU(inplace=True)]
        conv3 = [nn.Conv3d(conv2_channels, out_channels=conv3_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(conv3_channels), nn.ReLU(inplace=True)]
        conv4 = [nn.Conv3d(conv3_channels, out_channels=conv4_channels, kernel_size=3, stride=1, padding=1), nn.BatchNorm3d(conv4_channels), nn.ReLU(inplace=True)]
        # pool1 = [nn.MaxPool3d(kernel_size=2, stride=2)]

        layers += conv1
        layers += conv2
        layers += conv3
        layers += conv4
        # layers += pool1

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        s = out.view(out.size(0), -1)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), F.tanh(v)

    def _get_conv_output(self, shape):
        bs = 1
        input_ = Variable(torch.rand(bs, *shape))
        output_feat = self.features(input_)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

if __name__ == '__main__':
    import copy
    import sys
    import numpy as np
    sys.path.append('..')

    from VoxelLogic import Board
    from VoxelGame import VoxelGame

    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

    n = 20
    x = 9
    y = 3
    z = 6

    args = dotdict({
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 15,
        'batch_size': 64,
        'cuda': True, #torch.cuda.is_available(),
        'num_channels': 512,
    })

    b = Board(x, y, z, n)
    g = VoxelGame(x, y, z, n)

    nnet = VoxelNNet(g, args)

    if args.cuda:
        nnet.cuda()

    board = b.pieces.copy()
    board = torch.FloatTensor(board.astype(np.float64))
    if args.cuda: 
        board = board.contiguous().cuda()

    board = Variable(board, volatile=True)
    board = board.view(1, 1, nnet.board_x, nnet.board_y, nnet.board_z)    

    nnet.eval()
    nnet.forward(board)