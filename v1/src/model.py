import torch
torch.manual_seed(123)
from torch.nn import Module, Conv2d, MaxPool2d, Linear, Dropout, BatchNorm2d, ReLU, LogSoftmax
import torch.nn.functional as F


class genreNet(Module):

    def __init__(self):
        super(genreNet, self).__init__()
        """
        TRY HIGHER LEARNING RATE + ADAPATIVE LEARNING RATE
        TRY MORE EPOCHS
        TRY 3 CONV LAYERS? KEEP SQUARE KERNEL SIZE
        TRY REDUCED AMOUNT OF CHANNELS? go from low to high to low?
        """

        self.conv1  = Conv2d(in_channels=1,     out_channels=64,    kernel_size=3,  stride=1,   padding=1)
        self.bn1    = BatchNorm2d(64)
        self.relu1  = ReLU()
        self.pool1  = MaxPool2d(kernel_size=2)

        self.conv2  = Conv2d(in_channels=64, out_channels=128,      kernel_size=3,  stride=1,   padding=1)
        self.bn2    = BatchNorm2d(128)
        self.relu2  = ReLU()
        self.pool2  = MaxPool2d(kernel_size=2)

        self.conv3  = Conv2d(in_channels=128, out_channels=256,      kernel_size=3,  stride=1,   padding=1)
        self.bn3    = BatchNorm2d(256)
        self.relu3  = ReLU()
        self.pool3  = MaxPool2d(kernel_size=4)

        self.conv4  = Conv2d(in_channels=256, out_channels=512,      kernel_size=3,  stride=1,   padding=1)
        self.bn4    = BatchNorm2d(512)
        self.relu4  = ReLU()
        self.pool4  = MaxPool2d(kernel_size=4)

        self.fc1    = Linear(in_features=2048,  out_features=1024)
        self.relu5  = ReLU()
        self.drop1  = Dropout(0.1)

        self.fc2    = Linear(in_features=1024,  out_features=256)
        self.relu6  = ReLU()
        self.drop2  = Dropout(0.1)

        self.fc3    = Linear(in_features=256,   out_features=10)

        self.log    = LogSoftmax(dim=-1)

    def forward(self, inp):
        # INPUT SHAPE: [ BATCH SIZE, 1, 128, 128 ]
        x   = self.relu1(self.bn1(self.conv1(inp)))
        x   = self.pool1(x)

        x   = self.relu2(self.bn2(self.conv2(x)))
        x   = self.pool2(x)

        x   = self.relu3(self.bn3(self.conv3(x)))
        x   = self.pool3(x)

        x   = self.relu4(self.bn4(self.conv4(x)))
        x   = self.pool4(x)

        # SHAPE: [ BATCH SIZE, 512, 2, 2 ]

        x   = x.view(x.size()[0], -1)
        x   = self.relu5(self.fc1(x))
        x   = self.drop1(x)

        x   = self.relu6(self.fc2(x))
        x   = self.drop2(x)

        x   = self.log(x)
        return x
