import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index], dtype=torch.float),
            torch.tensor(self.label[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, dilation, groups=1
    ):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            dilation=self.dilation,
            padding="same",  # realized that it was added in a newer version of pytorch
        )

    def forward(self, x):

        net = x

        # compute pad shape
        # in_dim = net.shape[-1]
        # out_dim = (in_dim + self.stride - 1) // self.stride
        # p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        # pad_left = p // 2
        # pad_right = p - pad_left
        # net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size, stride):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size, stride=stride)

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, dilation, use_maxpool=True
    ):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.use_maxpool = use_maxpool

        # the first conv
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=self.dilation,
            stride=1,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        if self.use_maxpool:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=2, stride=2)

        self.do1 = nn.Dropout(p=0.3)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        out = self.conv1(out)
        # print('conv', out.shape)
        out = self.bn1(out)
        # print('bn', out.shape)
        out = self.relu1(out)
        # print('relu', out.shape)
        if self.use_maxpool:
            out = self.max_pool(out)
            # print('max_pool', out.shape)
        out = self.do1(out)
        # print('co', out.shape)
        return out


class DeepNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(DeepNet, self).__init__()

        self.block_list = nn.ModuleList()
        self.block_list.append(
            BasicBlock(
                in_channels=1,
                out_channels=320,
                kernel_size=24,
                dilation=1,
                use_maxpool=True,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=320,
                out_channels=256,
                kernel_size=16,
                dilation=4,
                use_maxpool=False,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=256,
                out_channels=256,
                kernel_size=16,
                dilation=4,
                use_maxpool=False,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=256,
                out_channels=256,
                kernel_size=16,
                dilation=4,
                use_maxpool=False,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=256,
                out_channels=128,
                kernel_size=8,
                dilation=4,
                use_maxpool=True,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=8,
                dilation=6,
                use_maxpool=False,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=8,
                dilation=6,
                use_maxpool=False,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=8,
                dilation=6,
                use_maxpool=False,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=8,
                dilation=6,
                use_maxpool=False,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=8,
                dilation=8,
                use_maxpool=True,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=128,
                out_channels=64,
                kernel_size=8,
                dilation=8,
                use_maxpool=False,
            )
        )
        self.block_list.append(
            BasicBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=8,
                dilation=8,
                use_maxpool=False,
            )
        )

        self.dense = nn.Linear(64, n_classes)

    def forward(self, x):

        out = x

        for b in self.block_list:
            out = b(out)

        # print(out.shape)
        out = torch.mean(out, dim=-1)
        # print(out.shape)
        out = self.dense(out)

        return out
