"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "Conv",
    "SENet",
    "ARNet",
    "Bottleneck",
    "CSPBlock",
    "AR_CSPBlock",
    "ResNetBlock",
    "ResNetLayer",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """pad to 'same' shape outputs."""
    if d > 1:
        k = d*(k-1) + 1
    if p is None:
        p = k//2  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


class SENet(nn.Module):
    """SENet"""

    def __init__(self, channels, ratio=1):
        """Initializes the class and sets the basic configurations and instance variables required."""
        super(ARNet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//ratio, False),
            nn.SiLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Feature enhancement"""
        B, C, L = x.size()  # batch, channel, length
        avg = self.avg_pool(x).view([B, C])
        fc = self.fc(avg).view([B, C, 1])
        return x*fc


class ARNet(nn.Module):
    """ARSM"""

    def __init__(self, channels, ratio=1):
        """Initializes the class and sets the basic configurations and instance variables required."""
        super(ARNet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//ratio, False),
            nn.SiLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Feature enhancement and ranking"""
        B, C, L = x.size()  # batch, channel, length
        avg = self.avg_pool(x).view([B, C])
        fc = self.fc(avg).view([B, C, 1])
        x = x*fc
        indices = torch.argsort(fc, dim=1, descending=False).expand(B, C, L)
        x = torch.gather(x, dim=1, index=indices)
        return x


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2=None, shortcut=True, k=(3, 3), e=1):
        """Initializes a bottleneck module with given input/output channels, shortcut option,
        kernels, and expansion.
        """
        super().__init__()
        if c2 is None:
            c2 = c1
            
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c1, c_, k[1], 1)
        self.add = shortcut and c1==c2

    def forward(self, x):
        """'forward()' applies the network to imput data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CSPBlock(nn.Module):
    """CSP Bottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, downsample=False):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, downsample."""
        super().__init__()
        self.n = n
        c_ = int(c1/(n+1))
        self.m = nn.ModuleList(Bottleneck(c1-self.c_*(i+1), shortcut=shortcut) for i in range(n))
        self.conv = Conv(int((n+1)*(n+2)*self.c_/2), c2, 1, s=2 if downsample else 1)

    def forward(self, x):
        """Forward pass through the AR-CSPNet block"""
        y = []
        for brach in range(self.n):
            y.append(x)
            y1, y2 = torch.tensor_split(x, (self.c_,), dim=1)
            x = self.m[brach](y2)
        y.append(x)
        return self.conv(torch.cat(y, dim=1))


class AR_CSPBlock(nn.Module):
    """ARSM"""

    def __init__(self, c1, c2, n=1, shortcut=False, downsample=False):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, downsample."""
        super().__init__()
        self.n = n

        c_ = int(c1/(n+1))
        self.m = nn.ModuleList(Bottleneck(c1-self.c_*(i+1), shortcut=shortcut) for i in range(n))
        self.ar = nn.ModuleList(ARNet(c1-self.c_*j) for j in range(n))
        self.conv = Conv(int((n+1)*(n+2)*self.c_/2), c2, 1, s=2 if downsample else 1)

    def forward(self, x):
        """Forward pass through the AR-CSPNet block"""
        y = []
        for brach in range(self.n):
            x = self.ar[brach](x)
            y.append(x)
            y1, y2 = torch.tensor_split(x, (self.c_,), dim=1)
            x = self.m[brach](y2)
        y.append(x)
        return self.conv(torch.cat(y, dim=1))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)

