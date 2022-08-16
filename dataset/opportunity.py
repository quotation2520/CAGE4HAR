from dataset import *
from dataset.dataset_generator import HARDataGenerator


class OPPORTUNITY(HARDataGenerator):
    # Referred to https://github.com/nhammerla/deepHAR/blob/master/data/datareader.py
    def __init__(self, window_length=76, clean=False, include_null=True):
        super(OPPORTUNITY, self).__init__()
        self.datapath = 'data/OpportunityUCIDataset'
        self.sampling_rate = 30
        self.WINDOW_LENGTH = window_length
        self.STRIDE = self.WINDOW_LENGTH // 2
        self.clean = clean
        self.include_null = include_null
        self._read_opportunity(self.datapath)

    def _read_opportunity(self, datapath):
        files = {
            'train': [
                'S1-ADL1.dat',                'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat',                               'S2-Drill.dat',
                'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat',                               'S3-Drill.dat', 
                'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'
            ],
            'val': [
                'S1-ADL2.dat'
            ],
            'test': [
                'S2-ADL4.dat', 'S2-ADL5.dat',       
                'S3-ADL4.dat', 'S3-ADL5.dat'
            ]
        }

        self.label_map = [
            (0,      'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        label2id = {str(x[0]): i for i, x in enumerate(self.label_map)}

#        cols = [
#            38, 39, 40, 41, 42, 43, 44, 45, 46,     # BACK (acc, gyro, magnetic) 
#            51, 52, 53, 54, 55, 56, 57, 58, 59,     # Right Upper Arm
#            64, 65, 66, 67, 68, 69, 70, 71, 72,     # Right Lower Arm
#            77, 78, 79, 80, 81, 82, 83, 84, 85,     # Left Upper Arm
#            90, 91, 92, 93, 94, 95, 96, 97, 98,     # Left Lower Arm
#            103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, # Left Shoe (Eu, Nav, Body, AngVel{Body,/Nav}Frame, Compass)
#            119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, # Right shoe
#            250]    # Mid-level Labels
        cols = [38, 39, 40, 51, 52, 53, 64, 65, 66, 77, 78, 79, 90, 91, 92,     # acc
                41, 42, 43, 54, 55, 56, 67, 68, 69, 80, 81, 82, 93, 94, 95,     # gyro
                250]     # label
        cols = [x-1 for x in cols] # labels for 18 activities (including other)

        data = {dataset: self._read_opp_files(datapath, files[dataset], cols, label2id, overlap)
                for dataset, overlap in zip(('train', 'val', 'test'), (True, False, False))}

        self.train_data = data['train']['inputs']
        self.train_label = data['train']['targets']
        self.val_data = data['val']['inputs']
        self.val_label = data['val']['targets']
        self.test_data = data['test']['inputs']
        self.test_label = data['test']['targets']

    def _read_opp_files(self, datapath, filelist, cols, label2id, overlap):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            nancnt = 0
            print('reading file %d of %d' % (i+1, len(filelist)))
            indiv_data = []
            indiv_labels = []
            with open(datapath.rstrip('/') + '/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    for ind in cols:
                        elem.append(line[ind])
                    # we can skip lines that contain NaNs, as they occur in blocks at the start
                    # and end of the recordings.
                    if sum([x == 'NaN' for x in elem]) == 0:
                        if elem[-1] in label2id:
                            indiv_data.append([float(x) / 1000 for x in elem[:-1]])
                            indiv_labels.append(label2id[elem[-1]])
          #  indiv_data = butterworth_filter(np.asarray(indiv_data), self.sampling_rate)
            split_data, split_labels = self.split_windows(np.asarray(indiv_data), np.asarray(indiv_labels), overlap)
            data.append(split_data)
            labels.append(split_labels)
        return {'inputs': np.concatenate(data, axis=0), 'targets': np.concatenate(labels)}

if __name__ == "__main__":
    opp = OPPORTUNITY()
    opp.dataset_verbose()

    pass