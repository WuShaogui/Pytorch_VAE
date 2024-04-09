import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0),
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(mid_channels))
        self.convs = nn.Sequential(*layers)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        feat = self.convs(x)
        output = x + feat
        return self.conv(output)
