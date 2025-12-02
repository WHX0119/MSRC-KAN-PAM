from torch import nn

class WDCNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(WDCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channel, 16, kernel_size=64, stride=16, padding=24),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),)
        self.conv4 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2))
        self.conv5 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, stride=1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=2, stride=2),)
        self.fc1 = nn.Sequential(nn.Linear(64, 100),
                                nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, out_channel))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size()[0], -1)
        
        self.feature1 = x
        x1 = self.fc1(x)
        self.feature2 = x1
        out = self.fc2(x1)
        return out, out