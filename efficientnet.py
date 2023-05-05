import torch
import torch.nn.functional as F
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, cin, squeeze_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(cin, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, cin, 1)
    def forward(self, x):
        out = self.avgpool(x)
        out = F.silu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out * x

class MBConv(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, expansion, se_ratio=4):
        super().__init__()
        layers = []
        expanded_channels = expansion * cin
        if expansion > 1: # expansion (only needed if > 1)
            layers.append(nn.Sequential(nn.Conv2d(cin, expanded_channels, 1, bias=False),
                                        nn.BatchNorm2d(expanded_channels),
                                        nn.SiLU()))

        # depthwise conv
        layers.append(nn.Sequential(nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride=stride, 
                                    padding=(kernel_size-1)//2, groups=expanded_channels, bias=False),
                                    nn.BatchNorm2d(expanded_channels),
                                    nn.SiLU()))
        
        # squeeze and excitation
        squeeze_channels = cin // se_ratio
        layers.append(SqueezeExcitation(expanded_channels, squeeze_channels))
        
        # reduction
        layers.append(nn.Sequential(nn.Conv2d(expanded_channels, cout, 1, bias=False),
                                    nn.BatchNorm2d(cout)))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class EfficientNetB0(nn.Module):
    def __init__(self):
        super().__init__()
        # input resolution, channels, layers, kernel size
        self.config = [
            (224, 32, 1, 3), 
            (112, 16, 1, 3),
            (112, 24, 2, 3),
            (56, 40, 2, 5),
            (28, 80, 3, 3),
            (14, 112, 3, 5),
            (14, 192, 4, 5),
            (7, 320, 1, 3),
            (7, 1280, 1, 1)
        ]
        
        features = []
        # setup stage 1 manually since it uses regular conv 
        cout = self.config[0][1]
        features += [nn.Sequential(
            nn.Conv2d(3, cout, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU())
            ]

        expansion = 1
        for i in range(1, len(self.config)-1):
            resolution, cout, layers, kernel_size = self.config[i]
            cin = self.config[i-1][1]
            next_resolution = self.config[i+1][0]

            stride = resolution // next_resolution
            features += self.make_mbconv_layers(cin, cout, kernel_size, 
                                                     stride=stride, expansion=expansion, num_layers=layers)
            expansion = 6 # after first MBConv, expansion is 6
        
        # add 1x1 conv at end
        cin, cout = self.config[-2][1], self.config[-1][1]
        features += [nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU())
            ]
        
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cout, 1000)
        )
    
    def make_mbconv_layers(self, cin, cout, kernel_size, stride=1, expansion=6, num_layers=1):
        layers = []
        for i in range(num_layers):
            layers.append(MBConv(cin, cout, kernel_size, stride, expansion))
            stride = 1 # set stride = 1 after first block in layer
            cin = cout
        return layers

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(self.avgpool(out), start_dim=1)
        out = self.classifier(out)
        return out