import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    自定义数据集类，用于加载数据和标签
    """
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
    扩展 nn.Conv1d 以支持 SAME 填充
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

    def forward(self, x):
        # 计算填充大小
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    扩展 nn.MaxPool1d 以支持 SAME 填充
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        # 计算填充大小
        net = x
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
    ResNet 基本块
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        downsample,
        use_bn,
        use_do,
        is_first_block=False,
    ):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # 第一个卷积层
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

        # 第二个卷积层
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups,
        )

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x

        # 第一个卷积层
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # 第二个卷积层
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # 如果需要下采样，也对 identity 进行下采样
        if self.downsample:
            identity = self.max_pool(identity)

        # 如果需要扩展通道，也对 identity 进行填充
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity
        return out


class ResNet1D(nn.Module):
    """
    ResNet1D 模型

    输入:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    输出:
        out: (n_samples)

    参数:
        in_channels: 输入维度，与 n_channel 相同
        base_filters: 第一个卷积层的滤波器数量，每 4 层翻倍
        kernel_size: 卷积核宽度
        stride: 卷积核移动步幅
        groups: 设置为 1 以实现 ResNeXt
        n_block: 块的数量
        n_classes: 类别数量
    """
    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        stride,
        groups,
        n_block,
        n_classes,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        verbose=False,
        classifier=None,
    ):
        super(ResNet1D, self).__init__()

        if classifier is None:
            classifier = linear_classifier(
                base_filters * 2 ** (n_block // increasefilter_gap - 1), n_classes
            )

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap  # 基础模型为 2
        self.increasefilter_gap = increasefilter_gap  # 基础模型为 4

        self.first_block_conv = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                in_channels = int(
                    base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap)
                )
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block,
            )
            self.basicblock_list.append(tmp_block)

        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.classifier = classifier

    def forward(self, x):
        out = x

        if self.verbose:
            print("input shape", out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print("after first conv", out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                pass
            out = net(out)
            if self.verbose:
                print(out.shape)

        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = self.classifier.forward(out)
        return out


class linear_classifier(nn.Module):
    """
    线性分类器

    输入:
        X: (batch_size, n_channel, n_length)

    输出:
        out: (batch_size, n_classes)

    参数:
        n_classes: 类别数量
    """
    def __init__(self, in_channels, n_classes, verbose=False):
        super(linear_classifier, self).__init__()

        self.in_channels = in_channels
        self.verbose = verbose
        self.do = nn.Dropout(0.5)
        self.dense = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        out = x
        if self.verbose:
            print("input shape", out.shape)

        out = out.mean(-1)
        if self.verbose:
            print("mean shape", out.shape)

        out = self.do(out)
        if self.verbose:
            print("mean shape", out.shape)

        out = self.dense(out)
        if self.verbose:
            print("out shape", out.shape)

        return out


class linear_feature_extractor_classifier(nn.Module):
    """
    线性特征提取分类器

    输入:
        X: (batch_size, n_channel, n_length)

    输出:
        out: (batch_size, n_classes)

    参数:
        n_classes: 类别数量
    """
    def __init__(self, in_channels, n_classes, verbose=False):
        super(linear_feature_extractor_classifier, self).__init__()

        self.in_channels = in_channels
        self.verbose = verbose
        self.do = nn.Dropout(0.5)
        self.f_layer = nn.Linear(in_channels, 32)
        self.f_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(32, n_classes)

    def forward(self, x):
        out = x
        if self.verbose:
            print("input shape", out.shape)

        out = out.mean(-1)
        if self.verbose:
            print("mean shape", out.shape)

        out = self.do(out)
        if self.verbose:
            print("mean shape", out.shape)

        out = self.f_layer(out)
        if self.verbose:
            print("out shape", out.shape)

        out = self.f_relu(out)
        if self.verbose:
            print("relu")

        out = self.dense(out)
        if self.verbose:
            print("out shape", out.shape)
        return out


class RNN_feature_extractor_classifier(nn.Module):
    """
    RNN 特征提取分类器

    输入:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    输出:
        out: (n_samples)

    参数:
        n_classes: 类别数量
    """
    def __init__(self, input_size, hidden_size, n_classes, num_layers, verbose=False):
        super(RNN_feature_extractor_classifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.verbose = verbose
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.drop = nn.Dropout(0.5)
        self.f_layer = nn.Linear(2 * hidden_size, 32)
        self.f_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(32, n_classes)

    def forward(self, x):
        out = x
        if self.verbose:
            print(out.shape)

        out = torch.permute(out, (0, 2, 1))

        if self.verbose:
            print("input lstm", out.shape)
        _, (out, _) = self.rnn(out)
        if self.verbose:
            print("lstm out", out.shape)

        out = self.drop(out)

        out_fw, out_bw = torch.split(out, self.num_layers, dim=0)
        out_fw = torch.mean(out_fw, dim=0)
        out_bw = torch.mean(out_bw, dim=0)
        out = torch.cat([out_fw, out_bw], dim=1)

        if self.verbose:
            print("reshaping", out.shape)

        out = self.f_layer(out)
        if self.verbose:
            print("out shape", out.shape)

        out = self.f_relu(out)
        if self.verbose:
            print("relu")

        out = self.dense(out)
        if self.verbose:
            print("out shape", out.shape)

        return out


class RNN_classifier(nn.Module):
    """
    RNN 分类器

    输入:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    输出:
        out: (n_samples)

    参数:
        n_classes: 类别数量
    """
    def __init__(self, input_size, hidden_size, n_classes, num_layers, verbose=False):
        super(RNN_classifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.verbose = verbose
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.drop = nn.Dropout(0.5)
        self.dense = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, x):
        out = x
        if self.verbose:
            print(out.shape)

        out = torch.permute(out, (0, 2, 1))

        if self.verbose:
            print("input lstm", out.shape)
        _, (out, _) = self.rnn(out)
        if self.verbose:
            print("lstm out", out.shape)

        out = self.drop(out)

        out_fw, out_bw = torch.split(out, self.num_layers, dim=0)
        out_fw = torch.mean(out_fw, dim=0)
        out_bw = torch.mean(out_bw, dim=0)
        out = torch.cat([out_fw, out_bw], dim=1)

        if self.verbose:
            print("reshaping", out.shape)

        out = self.dense(out)
        if self.verbose:
            print("out shape", out.shape)

        return out
