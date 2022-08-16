import torch.nn as nn
import torch.nn.functional as F

# Implementation of "LSTM-CNN Architecture for Human Activity Recognition" by Kun Xia et al. (IEEE Access, 2020)
class LSTMConvNet(nn.Module):
    def __init__(self, n_feat, n_cls):
        super(LSTMConvNet, self).__init__()
        self.n_feat = n_feat
        self.lstm1 = nn.LSTM(input_size=n_feat, hidden_size=32)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(128)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=n_cls)
        )

    def forward(self, x):
        bs, n_feat, window = x.shape
        x = x.permute([2, 0, 1])
        out, hidden = self.lstm1(x)
        out, hidden = self.lstm2(out, hidden)
        out = out.permute([1, 2, 0])
        out = self.conv1(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = out.mean(dim=-1).unsqueeze(dim=-1)
        out = self.bn(out)
        out = out.reshape(bs, -1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = LSTMConvNet(6, 6)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))