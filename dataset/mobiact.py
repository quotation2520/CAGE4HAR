from dataset import *
from dataset.dataset_generator import HARDataGenerator

class MobiAct(HARDataGenerator):
    def __init__(self, window_length=128, clean=False, fall=True):
        super(MobiAct, self).__init__()
        self.clean = clean
        self.fall = fall
        self.sampling_rate = 200
        self.WINDOW_LENGTH = window_length
        self.STRIDE = self.WINDOW_LENGTH // 2
        subject_ids = list(range(1, 68))    # 67 subjects
        train_split = subject_ids[:47]      # 47 subjects
        val_split = subject_ids[47:57]      # 10 subjects
        test_split = subject_ids[57:67]     # 10 subjects
        self.datapath = "data/MobiAct_Dataset_v2.0"
        self.label_map = [
            # ADLs
            ('STD', 'Standing'),
            ('WAL', 'Walking'),
            ('JOG', 'Jogging'),
            ('JUM', 'Jumping'),
            ('STU', 'Stairs up'),
            ('STN', 'Stairs down'),
            ('SCH', 'Stand to sit(sit on chair)'),
            ('SIT', 'Sitting on chair'),
            ('CHU', 'Sit to stand(chair up)'),
            ('CSI', 'Car-step in'),
            ('CSO', 'Car-step out'),
            ('LYI', 'Lying'),
            # Falls
            ('FOL', 'Forward-lying'),
            ('FKL', 'Front-knees-lying'),
            ('BSC', 'Back-sitting-chair'),
            ('SDL', 'Sideward-lying'),
        ]
        self.label2id = {x[0]: i for i, x in enumerate(self.label_map)}

        self.train_data, self.train_label = self._read_data(train_split)
        self.val_data, self.val_label = self._read_data(val_split)
        self.test_data, self.test_label = self._read_data(test_split)


    def _read_data(self, split, downsample=True):
        data = []
        label = []
        for path, subdir, files in os.walk(self.datapath + "/Annotated Data"):
            for name in files:
                if int(name.split('_')[1]) in split:
                    print(name)
                    signal = pd.read_csv(os.path.join(path, name), header=0, comment=';')
                    data_tmp = np.asarray(signal[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']])
                    label_tmp = np.asarray(signal[['label']])
                    data_tmp = butterworth_filter(data_tmp, self.sampling_rate)
                    if downsample:      # 200Hz -> 50Hz
                        label_tmp = label_tmp[::4]
                        data_tmp = data_tmp[::4]
                    label_tmp = np.asarray([self.label2id[x.item()] for x in label_tmp], dtype=int)
                    split_data, split_label = self.split_windows(data_tmp, label_tmp)
                    if split_data is not None:
                        data.append(split_data)
                        label.append(split_label)

        data, label = np.concatenate(data, axis=0), np.concatenate(label)
        if not self.fall:
            data, label = data[label<12], label[label<12]
        return data, label

if __name__ == "__main__":
    mobiact = MobiAct()
    mobiact.dataset_verbose()

    pass