import numpy as np
import os
from torch.utils.data import Dataset
from collections import Counter
import torch

class HARDataset(Dataset):
    def __init__(self, dataset='UCI_HAR', split='train', window_width=0, include_null=True, clean=True, zeroone=False, include_fall=False, use_portion=1.0):
        self.no_fall = not include_fall
        self._select_dataset(dataset)
        if window_width != 0:
            self.window_width = window_width
        dir_name = 'splits'
        if dataset =='MobiAct' and not include_fall:
            dir_name = dir_name + '_Xfall' 
        if not clean:
            if dataset in ['PAMAP2', 'Opportunity', 'mHealth', 'MobiAct']: 
                dir_name = dir_name + '_Xclean'
        if not include_null:
            dir_name = dir_name + '_Xnull'
        data_path = os.path.join(self.ROOT_PATH, dir_name, split + "_X_{}.npy".format(self.window_width))
        label_path = os.path.join(self.ROOT_PATH, dir_name, split + "_Y_{}.npy".format(self.window_width))
        self.data = np.load(data_path)
        self.label = np.load(label_path)

        if use_portion < 1:
            data_len = int(use_portion * len(self.data))
            self.data = self.data[:data_len]
            self.label = self.label[:data_len]
        
        _, self.feat_dim, self.window_width = self.data.shape
        samples = self.data.transpose(1, 0, 2).reshape(self.feat_dim, -1)
        if split == 'train':
            self.mean = np.mean(samples, axis=1)
            self.std = np.std(samples, axis=1)
        self.zeroone = zeroone
        
    def normalize(self, mean, std):
        self.data = self.data - mean.reshape(1, -1, 1)
        self.data = self.data / std.reshape(1, -1, 1)
        
    def _select_dataset(self, dataset):
        if dataset == 'UCI_HAR':
            self.ROOT_PATH = "data/UCI HAR Dataset"
            self.sampling_rate = 50
            self.n_actions = 6
            self.window_width = 128
        elif dataset == 'USC_HAD':
            self.ROOT_PATH = "data/USC-HAD"
            self.sampling_rate = 50 # downsampled from 100
            self.n_actions = 12
            self.window_width = 128
        elif dataset == 'Opportunity':
            self.sampling_rate = 30
            self.n_actions = 18
            self.window_width = 76
            self.ROOT_PATH = "data/OpportunityUCIDataset"
        elif dataset == 'PAMAP2':
            self.ROOT_PATH = "data/PAMAP2_Dataset"
            self.sampling_rate = 50 # downsampled from 100
            self.n_actions = 13
            self.window_width = 128
        elif dataset == 'mHealth':
            self.ROOT_PATH = "data/MHEALTHDATASET"
            self.sampling_rate = 50 # downsampled from 100
            self.n_actions = 13
            self.window_width = 128
        elif dataset == 'MobiAct':
            self.ROOT_PATH = "data/MobiAct_Dataset_v2.0"
            self.sampling_rate = 50 # downsampled from 200
            self.n_actions = 16
            if self.no_fall:
                self.n_actions = 12
            self.window_width = 128
            
        elif dataset == 'mmHAD':
            self.ROOT_PATH = "data/multi_modal_sensor_hardness_dataset/data_annotated"
            self.sampling_rate = 20
            self.n_actions = 7
            self.window_width = 52
        else:
            raise NotImplementedError("Dataset not supported")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.zeroone:
            data = torch.Tensor(self.data[i])
            data = data - data.min(dim=1).values.unsqueeze(dim=1) / (data.max(dim=1).values - data.min(dim=1).values).unsqueeze(dim=1)
            return data, self.label[i]
        return torch.Tensor(self.data[i]), self.label[i]

if __name__ == "__main__":
    ds = 'PAMAP2'
    train = HARDataset(dataset=ds, split="train", include_null=True, clean=False)
    val = HARDataset(dataset=ds, split="val", include_null=True, clean=False)
    test = HARDataset(dataset=ds, split="test", include_null=True, clean=False)
    print("# train : {}".format(len(train)))
    n_train = dict(Counter(train.label))
    print(sorted(n_train.items()))
    print("# val : {}".format(len(val)))
    n_val = dict(Counter(val.label))
    print(sorted(n_val.items()))
    print("# test : {}".format(len(test)))
    n_test = dict(Counter(test.label))
    print(sorted(n_test.items()))
    pass