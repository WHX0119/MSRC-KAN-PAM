import torch
import torch.nn as nn
from efficient_kan.kan import KANLinear

class MSRConv_block(nn.Module):
    def __init__(self, sample_length, in_channel, out_channel):
        super(MSRConv_block, self).__init__()
        self.scale_n = out_channel // in_channel
        self.mconv = nn.ModuleList([nn.Conv1d(in_channels = in_channel, out_channels = in_channel,
                                              kernel_size = 3+2*i,   # kernel_size = [3, 5, 7, 9]
                                              stride = 1,            # stride = 1
                                              padding = 1+i,         # padding = [1, 2, 3, 4]
                                              bias = False)
                                    for i in range(self.scale_n)])
        self.norm = nn.LayerNorm(sample_length)
        self.act = nn.Tanh()

    def forward(self, x):
        features = []
        for conv in self.mconv:
            features.append(self.norm(conv(x))+x)
        y = torch.cat(features, dim=1)    # [b, 1, 1024] -> [b, scale_n, 1024]
        return self.act(y)

def AdaptiveAbsAvgPool1d(x):
    return torch.abs(x).mean(2)

class MSRC_KAN_PAM(nn.Module):
    def __init__(self, sample_length, in_channel, conv_channels, n_class):
        super(MSRC_KAN_PAM, self).__init__()
        self.MSRConv_block = MSRConv_block
        self.feature_extraction_layers = []
        for _, conv_channel in enumerate(conv_channels):
            self.feature_extraction_layers.append(self.MSRConv_block(sample_length, in_channel, conv_channel))
            in_channel = conv_channel
        self.feature_extraction_layers = nn.Sequential(*self.feature_extraction_layers)

        # self.GAP = nn.AdaptiveAvgPool1d(1)
        # self.GMP = nn.AdaptiveMaxPool1d(1)
        self.AAAP = AdaptiveAbsAvgPool1d
        # self.fc = nn.Linear(conv_channels[-1], n_class)
        self.fc = KANLinear(in_features=conv_channels[-1], out_features=n_class)

    def forward(self, x):
        # [b, 1, 1024] => [b, channels, 1024]
        x_freq = self.feature_extraction_layers(x)
        # [b, channels, 1024] => [b, channels, 1024] => [b, channels]
        x_out = self.AAAP(x_freq)
        x_out = self.fc(x_out)      # [b, channels] => [b, n_class]
        return x_freq, x_out
