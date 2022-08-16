from dataset import *
from dataset.dataset_generator import HARDataGenerator


class USC_HAD(HARDataGenerator):
    def __init__(self, window_length=128, downsample=True):
        super(USC_HAD, self).__init__()
        self.datapath = "data/USC-HAD"
        self.sampling_rate = 100
        if downsample:
            self.sampling_rate = self.sampling_rate // 2
        self.WINDOW_LENGTH = window_length
        self.STRIDE = self.WINDOW_LENGTH // 2
        self._read_USC_HAD()

    def _read_dir(self):
        subject = []
        act_num = []
        sensor_readings = []
        for path, subdirs, files in os.walk(self.datapath):
            for name in files:
                if name.endswith('.mat'):
                    mat = scipy.io.loadmat(os.path.join(path, name))
                    subject.extend(mat['subject'])
                    signal = mat['sensor_readings']
                    signal = butterworth_filter(signal, self.sampling_rate)
                    sensor_readings.append(signal)

                    if mat.get('activity_number') is None:      # Subject13/a11t4
                        act_num.append(['11'])
                    else:
                        act_num.append(mat['activity_number'])
        return subject, act_num, sensor_readings

    def _read_USC_HAD(self, downsample=True):
        subject, act_num, sensor_readings = self._read_dir()

        train_id = [1,2,3,4,5,6,7,8,9,10]
        val_id = [11, 12]
        test_id = [13,14]

        self.train_data, self.train_label = [], []
        self.val_data, self.val_label = [], []
        self.test_data, self.test_label = [], []
        for i in range(840):
            if downsample:
                sensor_readings[i] = np.asarray(sensor_readings[i])[::2,:]

            if int(subject[i]) in train_id:
                self.split_data, self.split_label = self.split_windows(sensor_readings[i],
                                                                        np.repeat(act_num[i], sensor_readings[i].shape[0]))
                self.split_label = self.split_label.astype(np.uint8) -1
                self.train_data.append(self.split_data)
                self.train_label.append(self.split_label)
            elif int(subject[i]) in val_id:
                self.split_data, self.split_label = self.split_windows(sensor_readings[i],
                                                                        np.repeat(act_num[i], sensor_readings[i].shape[0]))
                self.split_label = self.split_label.astype(np.uint8) -1
                self.val_data.append(self.split_data)
                self.val_label.append(self.split_label)
            else:
                self.split_data, self.split_label = self.split_windows(sensor_readings[i],
                                                                       np.repeat(act_num[i], sensor_readings[i].shape[0]),
                                                                       overlap=False)
                self.split_label = self.split_label.astype(np.uint8) -1
                self.test_data.append(self.split_data)
                self.test_label.append(self.split_label)

        self.train_data = np.concatenate(self.train_data)
        self.train_label = np.concatenate(self.train_label)
        self.val_data = np.concatenate(self.val_data)
        self.val_label = np.concatenate(self.val_label)
        self.test_data = np.concatenate(self.test_data)
        self.test_label = np.concatenate(self.test_label)

if __name__ == "__main__":
    usc = USC_HAD()
    usc.dataset_verbose(usc)

    pass