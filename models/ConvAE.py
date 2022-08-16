import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

# Implementation of "On the role of features in human activity recognition."  by Harish Haresamudram et al. (ISWC, 2019)
# Referred https://github.com/harkash/on-the-role-of-features-in-har

class Vggish(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, last=False, stride=1):
        """
        Initializing the structure of the basic VGG-like block
        :param in_channels: the number of input channels
        :param out_channels: the number of output channels
        :param pool: flag to perform max pooling in the end
        :param last: flag to check if the block is the last layer in the autoencoder
        :param stride: stride for the convolutional layers
        """
        super(Vggish, self).__init__()
        # The two convolutional layers
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                stride=(stride, stride), padding=(1, 1), bias=False)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                                stride=(stride, stride), padding=(1, 1), bias=False)

        # Batch normalization
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        # Flags
        self.last = last
        self.pool = pool

        # Setting up the max pooling
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), return_indices=True)

        # Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
                init.xavier_normal_(m.weight)
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, inputs):
        """
        Performing the forward pass
        :param inputs: the actual data from the batch
        :return: output after the forward pass
        """
        x = F.relu(self.bn_1(self.conv_1(inputs)))

        # If it is the last layer, use sigmoid activation instead of hyperbolic tangent
        if self.last:
            x = torch.tanh(self.bn_2(self.conv_2(x)))
        else:
            x = F.relu(self.bn_2(self.conv_2(x)))

        # Performing max pooling if needed
        if self.pool:
            x, indices = self.max_pool(x)

        return x
# ----------------------------------------------------------------------------------------------------------------------

class MLP_Classifier(nn.Module):
    def __init__(self, n_cls):
        super(MLP_Classifier, self).__init__()
        self.linear_1 = nn.Linear(in_features=100, out_features=2048)
        self.linear_2 = nn.Linear(in_features=2048, out_features=512)
        self.out = nn.Linear(in_features=512, out_features=n_cls)

        self.bn_1 = nn.BatchNorm1d(2048)
        self.bn_2 = nn.BatchNorm1d(512)

        self.dropout_1 = nn.Dropout(0.4)
        self.dropout_2 = nn.Dropout(0.4)

        # Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
                init.xavier_normal_(m.weight)
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, inputs):
        linear_1 = self.dropout_1(F.relu(self.bn_1(self.linear_1(inputs))))
        linear_2 = self.dropout_2(F.relu(self.bn_2(self.linear_2(linear_1))))
        out = self.out(linear_2)

        return out


class CAE(nn.Module):
    def __init__(self, n_feat, latent_size=100):
        """
        Creating a convolutional autoencoder model
        :param dataset: choice of dataset. E.g.: Opportunity, Skoda etc
        :param n_feat: number of sensors
        :param latent_size: latent representation size
        """
        super(CAE, self).__init__()
        # Convolution
        self.up_conv_1 = Vggish(in_channels=1, out_channels=64)
        self.up_conv_2 = Vggish(in_channels=64, out_channels=128)
        self.up_conv_3 = Vggish(in_channels=128, out_channels=256)
        self.up_conv_4 = Vggish(in_channels=256, out_channels=512)

        # Flattening
        self.embedding = nn.Linear(512 * 8 * n_feat, latent_size)
        self.de_embedding = nn.Linear(latent_size, 512 * 8 * n_feat)
        self.bn_1 = nn.BatchNorm1d(latent_size)
        self.bn_2 = nn.BatchNorm1d(512 * 8 * n_feat)

        # Deconvolution
        self.down_conv_4 = Vggish(in_channels=512, out_channels=256, pool=False)
        self.down_conv_3 = Vggish(in_channels=256, out_channels=128, pool=False)
        self.down_conv_2 = Vggish(in_channels=128, out_channels=64, pool=False)
        self.down_conv_1 = Vggish(in_channels=64, out_channels=1, pool=False, last=True)


        # Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
                init.xavier_normal_(m.weight)
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, inputs):
        conv_1 = self.up_conv_1(inputs)  # 64x12x38
        conv_2 = self.up_conv_2(conv_1)  # 128x6x19
        conv_3 = self.up_conv_3(conv_2)  # 256x3x9
        conv_4 = self.up_conv_4(conv_3)  # 512x1x4

        # Vectorizing conv_4 and reducing size to the bottleneck
        rep = conv_4.view(conv_4.shape[0], -1)
        embedding_out = self.embedding(rep)
        embedding = F.relu(self.bn_1(embedding_out))
        de_embedding = F.relu(self.bn_2(self.de_embedding(embedding)))
        conv_back = de_embedding.view(conv_4.shape)

        pad_4 = F.interpolate(conv_back, scale_factor=(1, 2), mode='nearest')
        de_conv_4 = self.down_conv_4(pad_4)
        pad_3 = F.interpolate(de_conv_4, scale_factor=(1, 2), mode='nearest')
        de_conv_3 = self.down_conv_3(pad_3)
        pad_2 = F.interpolate(de_conv_3, scale_factor=(1, 2), mode='nearest')
        de_conv_2 = self.down_conv_2(pad_2)
        pad_1 = F.interpolate(de_conv_2, scale_factor=(1, 2), mode='nearest')
        de_conv_1 = self.down_conv_1(pad_1)
        
        return de_conv_1, embedding_out