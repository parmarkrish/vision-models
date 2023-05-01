import torch.nn as nn
import torch.nn.functional as F

import copy

def conv_bn(c_in, c_out, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm2d(c_out))

class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn = conv_bn(3, 64, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
    def forward(self, x):
        out = self.conv_bn(x)
        out = F.relu(self.maxpool(out))
        return out

class BasicBlock(nn.Module):
    def __init__(self, channels, mode=None):
        super().__init__()
        # for this block, mode = None and mode = 'after_stem' will have the same behavior
        assert mode in [None, 'downsample', 'after_stem']
        self.mode = mode
        if self.mode == 'downsample':
            self.conv_bn1 = conv_bn(channels // 2, channels, 3, padding=1, stride=2)
            self.proj = conv_bn(channels // 2, channels, 1, stride=2)
        else:
            self.conv_bn1 = conv_bn(channels, channels, 3, padding=1)
        self.conv_bn2 = conv_bn(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv_bn1(x))
        out = self.conv_bn2(out)
        if self.mode == 'downsample':
            x = self.proj(x)
        out = F.relu(x + out)
        return out

class BottleneckBlock(nn.Module):
    def __init__(self, channels, mode=None, expansion_ratio=4):
        super().__init__()
        assert mode in [None, 'downsample', 'after_stem']
        self.mode = mode
        if mode == 'downsample':
            self.conv_bn1 = conv_bn(channels * expansion_ratio // 2, channels, 1, stride=2)
            self.proj = conv_bn(channels * expansion_ratio // 2, channels * expansion_ratio, 1, stride=2)
        elif mode == 'after_stem':
            self.conv_bn1 = conv_bn(channels, channels, 1)
            self.proj = conv_bn(channels, channels * expansion_ratio, 1)
        else:
            self.conv_bn1 = conv_bn(channels * expansion_ratio, channels, 1)
        self.conv_bn2 = conv_bn(channels, channels, 3, padding=1)
        self.conv_bn3 = conv_bn(channels, channels * expansion_ratio, 1)
    def forward(self, x):
        out = F.relu(self.conv_bn1(x))
        out = F.relu(self.conv_bn2(out))
        out = self.conv_bn3(out)
        if self.mode: # add projection if mode == 'downsample' or 'after_stem'
            x = self.proj(x)
        out = F.relu(x + out)
        return out

class Head(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.avgpool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(c_in, 1000)
    def forward(self, x):
        return self.linear(self.flatten(self.avgpool(x)))

def create_stages(channels, mode, use_bottleneck, N):
    Block = BottleneckBlock if use_bottleneck else BasicBlock
    downsample_block = []
    if mode:
        downsample_block.append(Block(channels, mode))
        N -= 1
    regular_blocks = [copy.deepcopy(Block(channels)) for _ in range(N)]
    return nn.Sequential(*(downsample_block + regular_blocks))

class ResNet(nn.Module):
    def __init__(self, res_blocks_per_stage, use_bottleneck=False):
        super().__init__()
        self.stem = Stem()

        stages = []
        channels_per_stage = 64
        mode = 'after_stem' # set inital mode
        for N in res_blocks_per_stage:
            stages.append(create_stages(channels_per_stage, mode, use_bottleneck, N))
            channels_per_stage *= 2 
            mode = 'downsample'
        self.stages = nn.Sequential(*stages)
        in_features = channels_per_stage * 2 if use_bottleneck else channels_per_stage // 2
        self.head = Head(in_features)
    
    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def create_resnet18():
    return ResNet([2, 2, 2, 2])

def create_resnet34():
    return ResNet([3, 4, 6, 3])

def create_resnet50():
    return ResNet([3, 4, 6, 3], use_bottleneck=True)

def create_resnet101():
    return ResNet([3, 4, 23, 3], use_bottleneck=True)

def create_resnet152():
    return ResNet([3, 8, 36, 3], use_bottleneck=True)