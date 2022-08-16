import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_feat, out_channels=out_feat, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_feat, out_channels=out_feat, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_feat, out_channels=out_feat, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out.mean(dim=-1)



class CAGE(nn.Module):
    def __init__(self, n_feat, n_cls, proj_dim=0):
        super(CAGE, self).__init__()
        self.proj_dim = proj_dim
        self.enc_A = Encoder(n_feat, 64)
        self.enc_G = Encoder(n_feat, 64)

        if self.proj_dim > 0:
            self.proj_A = nn.Linear(in_features=64, out_features=proj_dim, bias=False)
            self.proj_G = nn.Linear(in_features=64, out_features=proj_dim, bias=False)
        
        self.temperature = nn.Parameter(torch.tensor([0.07]), requires_grad=True)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=n_cls)
        )


    def forward(self, x_accel, x_gyro, return_feat=False):
        f_accel = self.enc_A(x_accel)
        f_gyro = self.enc_G(x_gyro)

        out =  self.classifier(torch.cat((f_accel, f_gyro), dim=-1))
        if self.proj_dim > 0:
            e_accel = self.proj_A(f_accel)
            e_gyro = self.proj_G(f_gyro)
        else:
            e_accel = f_accel
            e_gyro = f_gyro
            
        logits = torch.mm(F.normalize(e_accel), F.normalize(e_gyro).T) * torch.exp(self.temperature)

        if return_feat:
            return logits, out, (F.normalize(e_accel), F.normalize(e_gyro))
        return logits, out
    
    def freeze_enc(self):
        for m in self._modules:
            if m != 'classifier':
                for p in self._modules[m].parameters():
                    p.requires_grad = False

             
class CAGE_EarlyFusion(nn.Module):
    def __init__(self, n_feat, n_cls):
        super(CAGE_EarlyFusion, self).__init__()
        self.encoder = Encoder(n_feat, 128)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=n_cls)
        )
        
    def forward(self, x, return_feat=False):
        out = self.encoder(x)
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    model = CGAP_EarlyFusion(3, 6)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

if __name__ == "__main__":
    model = CAGE(3, 6, 64)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))