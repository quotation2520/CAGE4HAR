import torch.nn as nn
import torch.nn.functional as F


class Baseline_CNN(nn.Module):
    def __init__(self, n_feat, n_cls, fc_ch):
        super(Baseline_CNN, self).__init__()
        self.n_feat = n_feat
        self.n_cls = n_cls
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.n_feat, out_channels=64, kernel_size=5),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=64 * (fc_ch - 16), out_features=128),   # in_feature 64 * 112 for UCI-HAR, WISDM  / 64 * 128 - 16 for opportunity
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=n_cls)
        )
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)
        self.fc3.apply(weights_init)

    def forward(self, x):
        bs, n_feat, window = x.shape
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(bs, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)
        
if __name__ == "__main__":
    model = Baseline_CNN(6, 6, 128)
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))