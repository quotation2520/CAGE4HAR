from dataset import *
from dataset.dataset_generator import HARDataGenerator

class UCI_HAR(HARDataGenerator):
    def __init__(self):
        super(UCI_HAR, self).__init__()
        self.datapath = "data/UCI HAR Dataset"
        self.sampling_rate = 50
        self.WINDOW_LENGTH = 128
        self.STRIDE = self.WINDOW_LENGTH // 2
        data, labels, subject = self._read_UCI_HAR("train")
        train_idx = np.where(subject < 28) 
        val_idx = np.where(subject >= 28)   # subject 28, 29, 30 used for valiadation
        self.train_data, self.train_label = data[train_idx], labels[train_idx]
        self.val_data, self.val_label = data[val_idx], labels[val_idx]
        self.test_data, self.test_label, _ = self._read_UCI_HAR("test")
    
    def _read_UCI_HAR(self, split="train"):
        split_path = os.path.join(self.datapath, split)
        # get label
        label_path = os.path.join(split_path, "y_" + split + ".txt")
        label = np.loadtxt(label_path)
        subject_path = os.path.join(split_path, "subject_" + split + ".txt")
        subject = np.loadtxt(subject_path)

        self.label_map = [
            (1, 'WALKING'), 
            (2, 'WALKING_UPSTAIRS'), 
            (3, 'WALKING_DOWNSTAIRS'), 
            (4, 'SITTING'), 
            (5, 'STANDING'), 
            (6, 'LAYING'),
            ]
            
        # get time series data
        signal_path = os.path.join(split_path, "Inertial Signals")
        channel_files = os.listdir(signal_path)
        channel_files.sort()
        datalist = []
        for f in channel_files[:6]: # body_acc/gyro_x/y/z
            signal = np.loadtxt(os.path.join(signal_path, f))
            datalist.append(signal)
        data = np.stack(datalist, axis=1)

        return data, label - 1, subject

if __name__ == "__main__":
    uci = UCI_HAR()
    uci.dataset_verbose()

    pass