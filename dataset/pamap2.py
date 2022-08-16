from dataset import *
from dataset.dataset_generator import HARDataGenerator

class PAMAP2(HARDataGenerator):
    def __init__(self, window_length=128, downsample=True, clean=False, include_null=True):
        super(PAMAP2, self).__init__()
        self.datapath = "data/PAMAP2_Dataset"
        self.sampling_rate = 100
        if downsample:
            self.sampling_rate = self.sampling_rate // 2
        self.WINDOW_LENGTH = window_length
        self.STRIDE = self.WINDOW_LENGTH // 2
        self.clean = clean
        self.include_null = include_null
        self._read_Pamap2(self.datapath)

    def _read_Pamap2(self, datapath):
        files = {
            'train': ['subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
                         'subject107.dat', 'subject108.dat', 'subject109.dat'],
            'val': ['subject105.dat'],
            'test': ['subject106.dat']
        }
        self.label_map = [
            (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'Nordic walking'),
            # (9, 'watching TV'),
            # (10, 'computer work'),
            # (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            # (18, 'folding laundry'),
            # (19, 'house cleaning'),
            # (20, 'playing soccer'),
            (24, 'rope jumping')
        ]
        if not self.include_null:
            self.label_map = self.label_map[1:]

        label2id = {str(x[0]): i for i, x in enumerate(self.label_map)}
        cols = [
            5, 6, 7, 22, 23, 24, 39, 40, 41,            # accel_hand, chest, ankle
            11, 12, 13, 28, 29, 30, 45, 46, 47          # gyro_hand, chest, ankle
        ]
        cols = [x - 1 for x in cols]

        data = {dataset: self._read_Pamap2_Files(datapath, files[dataset], cols, label2id, overlap)
                for dataset, overlap in zip(('train', 'val', 'test'), (True, False, False))}

        self.train_data = data['train']['inputs']
        self.train_label = data['train']['targets']
        self.val_data = data['val']['inputs']
        self.val_label = data['val']['targets']
        self.test_data = data['test']['inputs']
        self.test_label = data['test']['targets']

    def _read_Pamap2_Files(self, datapath, filelist, cols, label2id, overlap, downsample=True):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            indiv_data = []
            indiv_labels = []
            with open(datapath.rstrip('/') + '/Protocol/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    # not including the non related activity
                    if line[1] not in label2id:
                        continue
                    for ind in cols:
                        elem.append(line[ind])
                    indiv_data.append([float(x) for x in elem[:]])
                    indiv_labels.append(label2id[line[1]])
                # interpolate nan values
                indiv_data = np.asarray(pd.DataFrame(indiv_data).interpolate(axis=0))
                indiv_data = butterworth_filter(indiv_data, self.sampling_rate)
            if len(np.argwhere(np.isnan(indiv_data))) > 0:
                print('still nan')
            if downsample:
                indiv_data = indiv_data[::2]
                indiv_labels = indiv_labels[::2]
            split_data, split_labels = self.split_windows(np.asarray(indiv_data), np.asarray(indiv_labels, dtype=int), overlap)
            data.append(split_data)
            labels.append(split_labels)
        return {'inputs': np.concatenate(data, axis=0), 'targets': np.concatenate(labels)}



if __name__ == "__main__":
    pamap2 = PAMAP2()
    pamap2.dataset_verbose()

    pass