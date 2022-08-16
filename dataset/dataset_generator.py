from dataset import *

class HARDataGenerator():
    def __init__(self):
        self.clean = False
        self.include_null = False


    def dataset_verbose(self):
        print(f"# train: {len(self.train_data)}")
        n_train = dict(Counter(self.train_label))
        print(sorted(n_train.items()))
        print(f"# val: {len(self.val_data)}")
        n_val = dict(Counter(self.val_label))
        print(sorted(n_val.items()))
        print(f"# test: {len(self.test_data)}")
        n_test = dict(Counter(self.test_label))
        print(sorted(n_test.items()))


    def save_split(self, folder_name='splits'):
        directory = '/'.join((self.datapath, folder_name))
        if not os.path.isdir(directory):
            os.makedirs(directory)
        np.save(directory + '/train_X_{}.npy'.format(self.WINDOW_LENGTH), self.train_data)
        np.save(directory + '/train_Y_{}.npy'.format(self.WINDOW_LENGTH), self.train_label)
        np.save(directory + '/val_X_{}.npy'.format(self.WINDOW_LENGTH), self.val_data)
        np.save(directory + '/val_Y_{}.npy'.format(self.WINDOW_LENGTH), self.val_label)
        np.save(directory + '/test_X_{}.npy'.format(self.WINDOW_LENGTH), self.test_data)
        np.save(directory + '/test_Y_{}.npy'.format(self.WINDOW_LENGTH), self.test_label)
        print(f'Split saved on {directory}')

    def split_windows(self, raw_data, raw_label, overlap=True):
        idx = 0
        endidx = len(raw_data)
        data = []
        label = []
        while idx < endidx - self.WINDOW_LENGTH:
            data_segment = raw_data[idx:idx+self.WINDOW_LENGTH].T
            if self.clean and len(np.unique(raw_label[idx:idx + self.WINDOW_LENGTH])) > 1:
                pass
            else:
                data.append(data_segment)
                label.append(raw_label[idx+self.WINDOW_LENGTH])
            if overlap:
                idx += self.STRIDE
            else:
                idx += self.WINDOW_LENGTH
        if len(data) == 0:
            return None, None
        return np.stack(data), np.asarray(label)