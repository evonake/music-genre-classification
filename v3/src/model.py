import sys

import torch
from torch.nn.modules.activation import LogSoftmax
from torch.nn import Module, Conv2d, ReLU, LSTM, Linear, Dropout, BatchNorm2d, MaxPool2d

torch.manual_seed(123)


class genreNet(Module):

    def __init__(self):
        super(genreNet, self).__init__()
        # input: 128x128 1e-2 (300) batch_size:16
        # t_loss : 0.0891  v_loss : 0.6970  t_acc : 98.07  v_acc : 84.90 test_acc: 84.70
        # * no plat: overfitting but v_acc increasing slowly
        # self.conv1    = Conv2d(in_channels=1,  out_channels=32, kernel_size=(20, 3), stride=1, padding=1)
        # self.bn1      = BatchNorm2d(32)
        # self.relu1    = ReLU()
        # self.pool1    = MaxPool2d(2)

        # self.conv2    = Conv2d(in_channels=32, out_channels=64, kernel_size=(20, 3), stride=1, padding=1)
        # self.bn2      = BatchNorm2d(64)
        # self.relu2    = ReLU()
        # self.pool2    = MaxPool2d(2)

        # self.conv3    = Conv2d(in_channels=64, out_channels=16, kernel_size=(20, 3), stride=1, padding=1)
        # self.bn3      = BatchNorm2d(16)
        # self.relu3    = ReLU()
        # self.pool3    = MaxPool2d(2)

        # self.rnn      = LSTM(input_size=16, hidden_size=20, num_layers=5, dropout=0.5, batch_first=True, bidirectional=False)

        # self.fc1      = Linear(in_features=20, out_features=10)
        # self.relu4    = ReLU()

        # self.log      = LogSoftmax(dim=-1)

        # try with bidirectinality? try less num_layers on lstm

        # input: 128x128 1e-2 (300) batch_size:16
        # t_loss : 0.0399 v_loss : 0.7619 t_acc : 99.13 v_acc : 85.10 (peak 86.00)
        # self.conv1    = Conv2d(in_channels=1,  out_channels=32, kernel_size=(20, 3), stride=1, padding=1)
        # self.bn1      = BatchNorm2d(32)
        # self.relu1    = ReLU()
        # self.pool1    = MaxPool2d(2)

        # self.conv2    = Conv2d(in_channels=32, out_channels=64, kernel_size=(20, 3), stride=1, padding=1)
        # self.bn2      = BatchNorm2d(64)
        # self.relu2    = ReLU()
        # self.pool2    = MaxPool2d(2)

        # self.conv3    = Conv2d(in_channels=64, out_channels=16, kernel_size=(20, 3), stride=1, padding=1)
        # self.bn3      = BatchNorm2d(16)
        # self.relu3    = ReLU()
        # self.pool3    = MaxPool2d(2)

        # self.rnn      = LSTM(input_size=16, hidden_size=20, num_layers=5, dropout=0.5, batch_first=True, bidirectional=True)

        # self.fc1      = Linear(in_features=40, out_features=10)
        # self.relu4    = ReLU()

        # self.log      = LogSoftmax(dim=-1)

        # does not work

        self.conv1    = Conv2d(in_channels=1,  out_channels=32, kernel_size=(20, 3), stride=(2, 1), padding=1)
        self.bn1      = BatchNorm2d(32)
        self.relu1    = ReLU()

        self.conv2    = Conv2d(in_channels=32, out_channels=64, kernel_size=(20, 3), stride=(2, 1), padding=1)
        self.bn2      = BatchNorm2d(64)
        self.relu2    = ReLU()

        self.conv3    = Conv2d(in_channels=64, out_channels=16, kernel_size=(20, 3), stride=(2, 1), padding=1)
        self.bn3      = BatchNorm2d(16)
        self.relu3    = ReLU()

        self.pool1    = MaxPool2d(2)

        self.rnn      = LSTM(input_size=16, hidden_size=20, num_layers=5, dropout=0.5, batch_first=True, bidirectional=True)

        self.fc1      = Linear(in_features=40, out_features=10)

        self.log      = LogSoftmax(dim=-1)

    def forward(self, inp):
        # INPUT MEANING: [ BATCH SIZE, CHANNELS, FEATURE VECTOR LENGTH, TIME ]
        # INPUT SHAPE:   [ BATCH SIZE, 1,        128,                   256 ]

        x       = self.relu1(self.bn1(self.conv1(inp)))
        # x       = self.pool1(x)
        # print('CON1:\t', x.shape, '\t--- --- --- --- --- ---')

        x       = self.relu2(self.bn2(self.conv2(x)))
        # x       = self.pool2(x)
        # print('CON2:\t', x.shape, '\t--- --- --- --- --- ---')

        x       = self.relu3(self.bn3(self.conv3(x)))
        # x       = self.pool3(x)
        # print('CON3:\t', x.shape, '\t--- --- --- --- --- ---')

        # x   = self.relu4(self.bn4(self.conv4(x)))
        # x   = self.pool4(x)
        # print('CON4:\t', x.shape, '\t--- --- --- --- --- ---')

        x       = self.pool1(x)
        # print('FINAL:\t', x.shape, '\t--- --- --- --- --- ---')
        out, (h_n, c_n)   = self.rnn(x[:, :, :, 0].view(x.shape[0], 1, x.shape[1]))
        for i in range(1, x.shape[3]):
            out, (h_n, c_n) = self.rnn(x[:, :, :, i].view(x.shape[0], 1, x.shape[1]), (h_n, c_n))
        # print('RNN:\t', out.shape, out.view(out.size()[0], -1).shape)

        x       = out.view(out.size()[0], -1)

        x       = self.fc1(x)

        x       = self.log(x)
        return x
