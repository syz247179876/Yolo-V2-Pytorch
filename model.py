"""
根据YOLO V1论文构建网络结构
"""
import typing as t
import torch
import torch.nn as nn
import torchvision.models as tv_model
from settings import *


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def max_pool_2x2(stride: int = 2):
    """2x2 max pool"""
    return nn.MaxPool2d((2, 2), stride=stride)


class BasicBlock(nn.Module):
    """
    基于Darknet-19的BasicBlock ==> ConvBnLeakReLu
    带有BN层和leaky relu激活函数
    BN: 用于加快模型收敛速度, 减少过拟合, 使得后一层无需总是学习前一层的分布
    leaky relu:
    1.用于解决梯度弥散问题, 因为其计算出的梯度值为0或1, 因此在反向传播导数连乘过程中, 梯度值不会像Sigmoid函数那样越来越小, 进而
    避免参数梯度更新慢的问题。
    2.单侧饱和区一定程度上可以减少噪声的干扰, 更具鲁棒性，比如负值(噪音)经过relu函数得到0, 避免了噪音的训练时的干扰
    3.使用leaky relu替代relu, 函数中在对负值区域由0 -> ax, a参数可学习, 解决当负值经过 leaky relu函数时, 也能计算出一个非0梯度值, 解决了
        出现死亡神经元问题。
    """

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: t.Tuple[int, int] = (1, 1),
            norm_layer: t.Optional[t.Callable[..., nn.Module]] = None,
    ):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer(out_planes) if norm_layer else nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(inplace=True)

        if kernel_size[0] == 1:
            self.conv = conv1x1(in_planes, out_planes)
        elif kernel_size[0] == 3:
            self.conv = conv3x3(in_planes, out_planes)
        else:
            raise Exception('not support kernel size except for 1x1, 3x3')

    def forward(self, inputs: torch.Tensor):
        _x = self.conv(inputs)
        _x = self.norm_layer(_x)
        _x = self.relu(_x)
        return _x


class Darknet19(nn.Module):
    def __init__(
            self,
            norm_layer: t.Optional[t.Callable[..., nn.Module]] = None,
    ):
        super(Darknet19, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d

        self.layer_1 = nn.Sequential(
            BasicBlock(3, 32, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.layer_2 = nn.Sequential(
            BasicBlock(32, 64, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.layer_3 = nn.Sequential(
            BasicBlock(64, 128, kernel_size=(3, 3)),
            BasicBlock(128, 64, kernel_size=(1, 1)),
            BasicBlock(64, 128, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.layer_4 = nn.Sequential(
            BasicBlock(128, 256, kernel_size=(3, 3)),
            BasicBlock(256, 128, kernel_size=(1, 1)),
            BasicBlock(128, 256, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.layer_5 = nn.Sequential(
            BasicBlock(256, 512, kernel_size=(3, 3)),
            BasicBlock(512, 256, kernel_size=(1, 1)),
            BasicBlock(256, 512, kernel_size=(3, 3)),
            BasicBlock(512, 256, kernel_size=(1, 1)),
            BasicBlock(256, 512, kernel_size=(3, 3)),
            nn.MaxPool2d((2, 2), stride=2),
        )
        self.layer_6 = nn.Sequential(
            BasicBlock(512, 1024, kernel_size=(3, 3)),
            BasicBlock(1024, 512, kernel_size=(1, 1)),
            BasicBlock(512, 1024, kernel_size=(3, 3)),
            BasicBlock(1024, 512, kernel_size=(1, 1)),
            BasicBlock(512, 1024, kernel_size=(3, 3)),
        )
        self.layer_7 = nn.Sequential(
            nn.Conv2d(1024, 1000, kernel_size=(1, 1)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Softmax(dim=0)  # 得到 1000x1x1 的张量
        )

    def forward(self, inputs: torch.Tensor):
        _x = self.layer_1(inputs)
        _x = self.layer_2(_x)
        _x = self.layer_3(_x)
        _x = self.layer_4(_x)
        _x = self.layer_5(_x)
        _x = self.layer_6(_x)
        _x = self.layer_7(_x)
        return _x


if __name__ == '__main__':
    model = Darknet19()
    test_inputs = torch.randn(5, 3, 224, 224)
    x = model(test_inputs)
    print(x.shape)
