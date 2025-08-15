import random
import torch
import numpy as np
from torch.utils.data import Dataset

class SignalLocalDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(int)

    def __getitem__(self, index):
        sig = self.signals[index]
        target = self.labels[index]
        return sig, target

    def __len__(self):
        return len(self.signals)

class FedBase:
    def create_signal_datasets(self, num_clients=8, num_pos=3):
        local_datasets = []
        test_datasets = []
        pos = ['00', '01', '10']
        for pos_id in range(num_pos):
            for client_id in range(num_clients):
                data = np.load(f"../Dataset_Drones/train_Power_Normalization/D{pos[pos_id]}_STFT/device{client_id + 1}_x_train.npy")
                label = np.load(f"../Dataset_Drones/train_Power_Normalization/D{pos[pos_id]}_STFT/device{client_id + 1}_y_train.npy")
                local_datasets.append(SignalLocalDataset(data, label))

            test_data = np.load(f"../Dataset_Drones/test_Power_Normalization/D{pos[pos_id]}_STFT/x_test.npy")
            test_label = np.load(f"../Dataset_Drones/test_Power_Normalization/D{pos[pos_id]}_STFT/y_test.npy")
            test_datasets.append(SignalLocalDataset(test_data, test_label))

        return local_datasets, test_datasets