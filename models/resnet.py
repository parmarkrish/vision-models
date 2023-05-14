import torch.nn as nn
import torch.nn.functional as F

def conv_bn(c_in, c_out, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False),
                         nn.BatchNorm2d(c_out))

class BasicBlock(nn.Module):
    expansion_ratio = 1
    def __init__(self, cin, channels, stride=1):
        super().__init__()
        assert stride <= 2, 'stride greater than 2 is not supported'
        self.need_proj = cin != channels or stride != 1
        self.conv_bn1 = conv_bn(cin, channels, 3, stride, padding=1)
        if self.need_proj:
            self.proj = conv_bn(cin, channels, 1, stride)
        self.conv_bn2 = conv_bn(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv_bn1(x))
        out = self.conv_bn2(out)
        if self.need_proj:
            x = self.proj(x)
        out = F.relu(x + out)
        return out

class BottleneckBlock(nn.Module):
    expansion_ratio = 4
    def __init__(self, cin, channels, stride=1):
        super().__init__()
        assert stride <= 2, 'stride greater than 2 is not supported'
        self.need_proj = cin != channels * self.expansion_ratio or stride != 1
        self.conv_bn1 = conv_bn(cin, channels, 1, stride=stride)
        if self.need_proj:
            self.proj = conv_bn(cin, channels * self.expansion_ratio, 1, stride=stride)
        self.conv_bn2 = conv_bn(channels, channels, 3, padding=1)
        self.conv_bn3 = conv_bn(channels, channels * self.expansion_ratio, 1)
    def forward(self, x):
        out = F.relu(self.conv_bn1(x))
        out = F.relu(self.conv_bn2(out))
        out = self.conv_bn3(out)
        if self.need_proj: 
            x = self.proj(x)
        out = F.relu(x + out)
        return out

class ResNet(nn.Module):
    def __init__(self, blocks_per_stage, use_bottleneck=False):
        super().__init__()
        num_stages = len(blocks_per_stage)
        self.cin = 64
        self.stem = nn.Sequential(conv_bn(3, 64, 7, stride=2, padding=3),
                                  nn.MaxPool2d(3, stride=2, padding=1),
                                  nn.ReLU())
        self.stages = nn.ModuleList()
        
        strides = [1] + [2 for _ in range(num_stages-1)] # only first stage has stride of 2
        channels_per_stage = 64
        for i in range(num_stages):
            self._create_stages(blocks_per_stage[i], channels_per_stage, strides[i], use_bottleneck)
            channels_per_stage *= 2

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(start_dim=1),
                                  nn.Linear(self.cin, 1000)) # cin will update as we create stages
    
    def _create_stages(self, num_blocks, channels, stride, use_bottleneck):
        Block = BottleneckBlock if use_bottleneck else BasicBlock
        blocks = []
        for _ in range(num_blocks):
            blocks.append(Block(self.cin, channels, stride))
            self.cin = channels * Block.expansion_ratio
            stride = 1 # only first block in a stage needs to downsample
        self.stages.append(nn.Sequential(*blocks))
    
    def forward(self, x):
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.head(out)
        return out

def resnet18():
    return ResNet([2, 2, 2, 2])

def resnet34():
    return ResNet([3, 4, 6, 3])

def resnet50():
    return ResNet([3, 4, 6, 3], use_bottleneck=True)

def resnet101():
    return ResNet([3, 4, 23, 3], use_bottleneck=True)

def resnet152():
    return ResNet([3, 8, 36, 3], use_bottleneck=True)