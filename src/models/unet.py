import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Double convolution block for U-Net
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, momentum=0.1, dropout_rate=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
        # Initialize weights
        for m in self.double_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.0):
        super().__init__()

        # if bilinear, use normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handling input sizes that are not perfectly divisible by 2
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                  diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Output convolution
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    U-Net architecture for medical image segmentation
    """
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256], bilinear=True, bn_momentum=0.1, dropout_rate=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.bn_momentum = bn_momentum
        self.dropout_rate = dropout_rate

        # Input layer
        self.inc = DoubleConv(in_channels, features[0], momentum=bn_momentum, dropout_rate=dropout_rate)
        
        # Downsampling path
        self.downs = nn.ModuleList()
        for i in range(len(features)-1):
            self.downs.append(Down(features[i], features[i+1], dropout_rate=dropout_rate))

        # Upsampling path
        self.ups = nn.ModuleList()
        prev_channels = features[-1]
        for feature in reversed(features[:-1]):
            self.ups.append(Up(prev_channels + feature, feature, bilinear, dropout_rate=dropout_rate))
            prev_channels = feature

        # Output layer
        self.outc = OutConv(features[0], out_channels)
        
        # Initialize output layer with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.outc.conv.weight)
        if self.outc.conv.bias is not None:
            nn.init.constant_(self.outc.conv.bias, 0)

    def forward(self, x):
        # Store intermediate outputs for skip connections
        x_downs = []
        
        # Downsampling path
        x = self.inc(x)
        x_downs.append(x)
        
        for down in self.downs:
            x = down(x)
            x_downs.append(x)

        # Upsampling path with skip connections
        x_downs = x_downs[:-1]  # Remove last element as it's the bottleneck
        for up, x_skip in zip(self.ups, reversed(x_downs)):
            x = up(x, x_skip)

        # Output layer
        return self.outc(x) 