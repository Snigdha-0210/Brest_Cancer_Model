import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv
from .nscgcn import NSCGCNModule

class UNetGCN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], num_nodes=64):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck
        bottleneck_channels = features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], bottleneck_channels)
        
        # GCN Module applied at the bottleneck
        self.gcn_module = NSCGCNModule(in_channels=bottleneck_channels, num_nodes=num_nodes)
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck + Graph Global Reasoning
        x = self.bottleneck(x)
        x = self.gcn_module(x)
        
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
