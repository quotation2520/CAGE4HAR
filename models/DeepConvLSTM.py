import torch.nn as nn
import torch.nn.functional as F

# Implementation of "Deep ConvLSTM with self-attention for human activity decoding using wearable sensors." by Satya P. Singh et al. (IEEE Sensors, 2020)
class DeepConvLSTM(nn.Module):
    def __init__(self, n_feat, n_cls):
        super(DeepConvLSTM, self).__init__()
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
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=n_cls)
        )
        
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        self.lstm1.apply(weights_init)
        self.lstm2.apply(weights_init)
        self.fc.apply(weights_init)

    def forward(self, x):
        bs, n_feat, window = x.shape
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.permute([2, 0, 1])
        out, hidden = self.lstm1(out)
        out, hidden = self.lstm2(out, hidden)
        out = out.permute([1, 2, 0])
        last = out[:,:,-1]
        out = self.fc(last)
        return out

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
             
if __name__ == "__main__":
    model = DeepConvLSTM(6, 6)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))