from dataset import *
from dataset.dataset_generator import HARDataGenerator


class MHEALTH(HARDataGenerator):
    def __init__(self, window_length=128, clean=False, include_null=True):
        super(MHEALTH, self).__init__()
        self.datapath = "data/MHEALTHDATASET"
        self.sampling_rate = 50
        self.WINDOW_LENGTH = window_length
        self.STRIDE = self.WINDOW_LENGTH // 2
        self.clean = clean
        self.include_null = include_null
        self._read_mHealth(self.datapath)

    def _read_mHealth(self, datapath):
        files = {
            'train': [1, 2, 3, 4, 5, 6, 7, 8],
            'val': [9],
            'test': [10]
        }
        self.label_map = [
            (0, 'null'),
            (1, 'Standing still'),
            (2, 'Sitting and relaxing'),
            (3, 'Lying down'),
            (4, 'Walking'),
            (5, 'Climbing stairs'),
            (6, 'Waist bends forward'),
            (7, 'Frontal elevation of arms'),
            (8, 'Knees bending (crouching)'),
            (9, 'Cycling'),
            (10, 'Jogging'),
            (11, 'Running'),
            (12, 'Jump front & back '),            
        ]
        if not self.include_null:
            self.label_map = self.label_map[1:]
        label2id = {str(x[0]): i for i, x in enumerate(self.label_map)}

        cols = [
            6, 7, 8, 15, 16, 17,            # accel (left ankle, right lower arm)
            9, 10, 11, 18, 19, 20,          # gyro (left ankle, right lower arm)
            24                              # label
        ]
        cols = [x - 1 for x in cols]

        data = {dataset: self._read_mHealth_Files(datapath, files[dataset], cols, label2id, overlap)
                for dataset, overlap in zip(('train', 'val', 'test'), (True, False, False))}

        self.train_data = data['train']['inputs']
        self.train_label = data['train']['targets']
        self.val_data = data['val']['inputs']
        self.val_label = data['val']['targets']
        self.test_data = data['test']['inputs']
        self.test_label = data['test']['targets']

    def _read_mHealth_Files(self, datapath, filelist, cols, label2id, overlap):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            indiv_data = []
            indiv_labels = []
            with open(datapath.rstrip('/') + '/mHealth_subject{}.log'.format(filename), 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])
                    signal = list(map(float, elem[:-1]))
                    label = elem[-1]
                    if label not in label2id:
                        continue
                    else:
                        indiv_data.append(signal)
                        indiv_labels.append(label2id[label])
            # interpolate nan values
                indiv_data = np.asarray(pd.DataFrame(indiv_data).interpolate(axis=0))
                indiv_data = butterworth_filter(indiv_data, self.sampling_rate)
            if len(np.argwhere(np.isnan(indiv_data))) > 0:
                print('still nan')
            split_data, split_labels = self.split_windows(np.asarray(indiv_data), np.asarray(indiv_labels, dtype=int), overlap)
            data.append(split_data)
            labels.append(split_labels)
        return {'inputs': np.concatenate(data, axis=0), 'targets': np.concatenate(labels)}

if __name__ == "__main__":
    mhealth = MHEALTH(clean=True)
    mhealth.dataset_verbose()

    pass