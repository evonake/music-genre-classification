import torch
from torch.nn.modules.activation import LogSoftmax
from torch.nn import Module, Conv2d, ReLU, LeakyReLU, Linear, Dropout, BatchNorm2d, MaxPool2d

torch.manual_seed(123)


class genreNet(Module):

    def __init__(self):
        super(genreNet, self).__init__()
        # 1e-3 (500); batch_size=16
        # train_loss: 0.4542; valid_loss: 0.6322; train_acc: 81.91; valid_acc: 85.80 (* plateau at 120, train_loss no plateau)
        # test acc: 86.20 431/500
        # self.conv1    = Conv2d(in_channels=1,    out_channels=64,    kernel_size=(64, 3),  stride=1)
        # self.bn1      = BatchNorm2d(64)
        # self.relu1    = ReLU()

        # self.conv2    = Conv2d(in_channels=64,    out_channels=16,    kernel_size=(64, 3),  stride=1)
        # self.bn2      = BatchNorm2d(16)
        # self.relu2    = ReLU()

        # self.pool1    = MaxPool2d(2)

        # self.fc1      = Linear(in_features=2016, out_features=128)
        # self.drop1    = Dropout(0.2)
        # self.relu4    = ReLU()

        # self.fc2      = Linear(in_features=128, out_features=10)
        # self.drop2    = Dropout(0.2)
        # self.relu5    = ReLU()

        # self.log      = LogSoftmax(dim=-1)


        # 1e-3 (500); batch_size = 16
        # train_loss: .0007; valid_loss: .6560; train_acc: 100.00; valid_acc: 86.00 (* plateau at 200, valid_loss rise from .60 (120) -> .6560 (500))
        # test acc: 88.4 442/500
        # self.conv1    = Conv2d(in_channels=1,    out_channels=128,    kernel_size=(64, 3),  stride=1)
        # self.bn1      = BatchNorm2d(128)
        # self.relu1    = ReLU()

        # self.conv2    = Conv2d(in_channels=128,    out_channels=16,    kernel_size=(64, 3),  stride=1)
        # self.bn2      = BatchNorm2d(16)
        # self.relu2    = ReLU()

        # self.pool1    = MaxPool2d(2)

        # self.fc1      = Linear(in_features=2016, out_features=128)
        # self.relu4    = ReLU()
        # self.drop1    = Dropout(0.2)

        # self.fc2      = Linear(in_features=128, out_features=10)
        # self.relu5    = ReLU()

        # self.log      = LogSoftmax(dim=-1)

        # 1e-3 (300) batch_size: 32
        # t_loss: .7288; v_loss: .5412;  t_acc: 72.34; v_acc: 87.80
        # * plateau at 125
        # test_acc: 90.80 454/500
        # self.conv1    = Conv2d(in_channels=1,  out_channels=8,   kernel_size=(34, 3), stride=1, padding=1)
        # self.bn1      = BatchNorm2d(8)
        # self.relu1    = ReLU()
        # self.pool1    = MaxPool2d(2)

        # self.conv2    = Conv2d(in_channels=8,  out_channels=64,  kernel_size=(24, 3), stride=1, padding=1)
        # self.bn2      = BatchNorm2d(64)
        # self.relu2    = ReLU()
        # self.pool2    = MaxPool2d(2)

        # self.conv3    = Conv2d(in_channels=64, out_channels=16, kernel_size=(14, 3),  stride=1, padding=1)
        # self.bn3      = BatchNorm2d(16)
        # self.relu3    = ReLU()
        # self.pool3    = MaxPool2d(2)

        # self.fc1      = Linear(in_features=512, out_features=64)
        # self.relu4    = ReLU()
        # self.drop1    = Dropout(0.3)

        # self.fc2      = Linear(in_features=64, out_features=10)
        # self.relu5    = ReLU()
        # self.drop2    = Dropout(0.3)

        # self.log      = LogSoftmax(dim=-1)

        # 1e-3 (300) lr has no effect; batch_size: 32
        # t_loss: .6075, v_loss: .4025; t_acc: 77.63; v_acc: 90.80
        # * acc plateau at 100; * loss plateau at 150
        # train_acc: 99.89 (3496 / 3500) ; valid_acc: 90.80 (908 / 1000) ; test_acc: 92.00 (460 / 500)
        # ^ very high train_acc (w/o dropout) -> increase dropout probabilities, decrease batch_size
        # self.conv1    = Conv2d(in_channels=1,  out_channels=32,   kernel_size=(34, 3), stride=1, padding=1)
        # self.bn1      = BatchNorm2d(32)
        # self.relu1    = ReLU()
        # self.pool1    = MaxPool2d(2)

        # self.conv2    = Conv2d(in_channels=32,  out_channels=64,  kernel_size=(24, 3), stride=1, padding=1)
        # self.bn2      = BatchNorm2d(64)
        # self.relu2    = ReLU()
        # self.pool2    = MaxPool2d(2)

        # self.conv3    = Conv2d(in_channels=64, out_channels=16, kernel_size=(14, 3),  stride=1, padding=1)
        # self.bn3      = BatchNorm2d(16)
        # self.relu3    = ReLU()
        # self.pool3    = MaxPool2d(2)

        # self.fc1      = Linear(in_features=512, out_features=64)
        # self.relu4    = ReLU()
        # self.drop1    = Dropout(0.25)

        # self.fc2      = Linear(in_features=64, out_features=10)
        # self.relu5    = ReLU()
        # self.drop2    = Dropout(0.25)

        # self.log      = LogSoftmax(dim=-1)

        # 1e-3 (200) batch_size: 32
        # t_loss: .6056; v_loss: .3934; t_acc: 77.17; v_acc: 91.20
        # * acc plateau at 120, * loss no plateau
        # train_acc: 99.89 (3496 / 3500) ; valid_acc: 91.40 (914 / 1000) ; test_acc: 91.80 (459 / 500)
        # self.conv1    = Conv2d(in_channels=1,  out_channels=32,   kernel_size=(34, 3), stride=1, padding=1)
        # self.bn1      = BatchNorm2d(32)
        # self.relu1    = ReLU()
        # self.pool1    = MaxPool2d(2)

        # self.conv2    = Conv2d(in_channels=32,  out_channels=128,  kernel_size=(24, 3), stride=1, padding=1)
        # self.bn2      = BatchNorm2d(128)
        # self.relu2    = ReLU()
        # self.pool2    = MaxPool2d(2)

        # self.conv3    = Conv2d(in_channels=128, out_channels=16, kernel_size=(14, 3),  stride=1, padding=1)
        # self.bn3      = BatchNorm2d(16)
        # self.relu3    = ReLU()
        # self.pool3    = MaxPool2d(2)

        # self.fc1      = Linear(in_features=512, out_features=64)
        # self.relu4    = ReLU()
        # self.drop1    = Dropout(0.25)

        # self.fc2      = Linear(in_features=64, out_features=10)
        # self.relu5    = ReLU()
        # self.drop2    = Dropout(0.25)

        # self.log      = LogSoftmax(dim=-1)

        # input size: 128x256, 1e-3 (150) -> 1e-4 (300) batch_size: 32
        # t_loss: .6168; v_loss: .3771; t_acc: 77.06; v_acc: 92.20
        # t_acc plat at 100; v_acc plateau at 200; t_loss plat at 135; v_loss plat at 195
        # train_acc: 99.69 (3489 / 3500) ; valid_acc: 92.20 (922 / 1000) ; test_acc: 92.20 (461 / 500)
        # --
        # input size: 128x128, 1e-3 (150) -> 1e-4 (300), batch_size: 32
        # * plat at 200
        # train_acc: 99.64 (6975 / 7000) ; valid_acc: 92.60 (1852 / 2000) ; test_acc: 91.40 (914 / 1000)
        # self.conv1    = Conv2d(in_channels=1,  out_channels=32,   kernel_size=(20, 3), stride=1, padding=1)
        # self.bn1      = BatchNorm2d(32)
        # self.relu1    = ReLU()
        # self.pool1    = MaxPool2d(2)

        # self.conv2    = Conv2d(in_channels=32,  out_channels=64,  kernel_size=(20, 3), stride=1, padding=1)
        # self.bn2      = BatchNorm2d(64)
        # self.relu2    = ReLU()
        # self.pool2    = MaxPool2d(2)

        # self.conv3    = Conv2d(in_channels=64, out_channels=16, kernel_size=(20, 3),  stride=1, padding=1)
        # self.bn3      = BatchNorm2d(16)
        # self.relu3    = ReLU()
        # self.pool3    = MaxPool2d(2)

        # self.fc1      = Linear(in_features=256, out_features=64)
        # self.relu4    = ReLU()
        # self.drop1    = Dropout(0.25)

        # self.fc2      = Linear(in_features=64, out_features=10)
        # self.relu5    = ReLU()
        # self.drop2    = Dropout(0.25)

        # self.log      = LogSoftmax(dim=-1)

        # batch_size: 10 128x128 1e-3 (150) -> 13-4 (300)
        # t_loss : 0.7378  v_loss : 0.3660  t_acc : 72.47  v_acc : 92.25
        # valid * plat at 200
        # train_acc: 99.50 (6965 / 7000) ; valid_acc: 92.20 (1844 / 2000) ; test_acc: 90.20 (902 / 1000)
        # self.conv1    = Conv2d(in_channels=1,  out_channels=32,   kernel_size=(20, 3), stride=1, padding=1)
        # self.bn1      = BatchNorm2d(32)
        # self.relu1    = ReLU()
        # self.pool1    = MaxPool2d(2)

        # self.conv2    = Conv2d(in_channels=32,  out_channels=64,  kernel_size=(20, 3), stride=1, padding=1)
        # self.bn2      = BatchNorm2d(64)
        # self.relu2    = ReLU()
        # self.pool2    = MaxPool2d(2)

        # self.conv3    = Conv2d(in_channels=64, out_channels=16, kernel_size=(20, 3),  stride=1, padding=1)
        # self.bn3      = BatchNorm2d(16)
        # self.relu3    = ReLU()
        # self.pool3    = MaxPool2d(2)

        # self.fc1      = Linear(in_features=256, out_features=64)
        # self.relu4    = ReLU()
        # self.drop1    = Dropout(0.35)

        # self.fc2      = Linear(in_features=64, out_features=10)
        # self.relu5    = ReLU()
        # self.drop2    = Dropout(0.35)

        # self.log      = LogSoftmax(dim=-1)

        # batch_size: 10 128x128 1e-3 (150) -> 13-4 (300)
        # t_loss : 0.0347  v_loss : 0.2669  t_acc : 99.47  v_acc : 91.80
        # train_acc: 99.89 (6992 / 7000) ; valid_acc: 91.80 (1836 / 2000) ; test_acc: 91.40 (914 / 1000)
        # self.conv1    = Conv2d(in_channels=1,  out_channels=32,   kernel_size=(20, 3), stride=1, padding=1)
        # self.bn1      = BatchNorm2d(32)
        # self.relu1    = ReLU()
        # self.pool1    = MaxPool2d(2)

        # self.conv2    = Conv2d(in_channels=32,  out_channels=64,  kernel_size=(20, 3), stride=1, padding=1)
        # self.bn2      = BatchNorm2d(64)
        # self.relu2    = ReLU()
        # self.pool2    = MaxPool2d(2)

        # self.conv3    = Conv2d(in_channels=64, out_channels=16, kernel_size=(20, 3),  stride=1, padding=1)
        # self.bn3      = BatchNorm2d(16)
        # self.relu3    = ReLU()
        # self.pool3    = MaxPool2d(2)

        # self.fc1      = Linear(in_features=256, out_features=64)
        # self.relu4    = ReLU()
        # self.drop1    = Dropout(0.45)

        # self.fc2      = Linear(in_features=64, out_features=10)
        # self.relu5    = ReLU()

        # self.log      = LogSoftmax(dim=-1)

        # test stride=(2, 1) with avgpool2d
        self.conv1    = Conv2d(in_channels=1,  out_channels=32,   kernel_size=(10, 3), stride=1, padding=1)
        self.bn1      = BatchNorm2d(32)
        self.relu1    = ReLU()
        self.pool1    = MaxPool2d(kernel_size=(3, 1))

        self.conv2    = Conv2d(in_channels=32,  out_channels=64,  kernel_size=(10, 3), stride=1, padding=1)
        self.bn2      = BatchNorm2d(64)
        self.relu2    = ReLU()
        self.pool2    = MaxPool2d(kernel_size=(3, 1))

        self.conv3    = Conv2d(in_channels=64, out_channels=16, kernel_size=(10, 3),  stride=1, padding=1)
        self.bn3      = BatchNorm2d(16)
        self.relu3    = ReLU()
        self.pool3    = MaxPool2d(kernel_size=(3, 1))

        self.fc1      = Linear(in_features=2048, out_features=128)
        self.relu4    = ReLU()
        self.drop1    = Dropout(0.25)

        self.fc2      = Linear(in_features=128, out_features=10)
        self.relu5    = ReLU()
        self.drop2    = Dropout(0.25)

        self.log      = LogSoftmax(dim=-1)

    def forward(self, inp):
        # INPUT MEANING: [ BATCH SIZE, CHANNELS, FEATURE VECTOR LENGTH, TIME ]
        # INPUT SHAPE:   [ BATCH SIZE, 1,        128,                   256 ]

        x   = self.relu1(self.bn1(self.conv1(inp)))
        x   = self.pool1(x)
        # print('CON1:\t', x.shape, '\t--- --- --- --- --- ---')

        x   = self.relu2(self.bn2(self.conv2(x)))
        x   = self.pool2(x)
        # print('CON2:\t', x.shape, '\t--- --- --- --- --- ---')

        x   = self.relu3(self.bn3(self.conv3(x)))
        x   = self.pool3(x)
        # print('CON3:\t', x.shape, '\t--- --- --- --- --- ---')

        # x   = self.relu4(self.bn4(self.conv4(x)))
        # x   = self.pool4(x)
        # print('CON4:\t', x.shape, '\t--- --- --- --- --- ---')

        # x     = self.pool1(x)
        # print('FINAL:\t', x.shape, '\t--- --- --- --- --- ---')
        x   = x.view(x.size()[0], -1)
        # print('reshape: ', x.shape)

        x   = self.fc1(x)
        x   = self.relu4(x)
        x   = self.drop1(x)

        x   = self.fc2(x)
        x   = self.relu5(x)
        x   = self.drop2(x)

        # x   = self.fc3(x)
        # x   = self.relu6(x)

        x   = self.log(x)
        # print(x.shape)
        return x
