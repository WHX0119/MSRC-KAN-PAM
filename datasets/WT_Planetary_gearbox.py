#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import os
from tqdm import tqdm

from datasets.add_noise import add_snr_noise

def data_load(args, file_path, label):
    data, lab = [], []
    signal = loadmat(file_path)['Data'].T[0].ravel()
    f1 = np.array(signal, dtype=np.float32).reshape(-1)

    if args.snr_noise:
        print('*****The data is added {}db noise*****'.format(args.snr), '\n')
        f1 = add_snr_noise(f1, args.snr)
    else:
        print('*****The data is not added noise*****\n')

    length = 500 * args.sample_length
    start, end = 0, args.sample_length
    while end <= length:
        sample = f1[start:end]    # <class 'numpy.ndarray'> (1024,) float32
        data.append(sample)       # <class 'list'> 500 -> <class 'numpy.ndarray'> (1024,) float32
        lab.append(label)         # <class 'list'> 500 -> <class 'int'>
        start += args.sample_length
        end += args.sample_length
    return data, lab

def dataset_save(args, save_name):
    data1, lab1 = [], []
    if args.D_num == 'D1':
        file_dir = './data/WT-Planetary Gearbox/20Hz'
    elif args.D_num == 'D2':
        file_dir = './data/WT-Planetary Gearbox/30Hz'
    elif args.D_num == 'D3':
        file_dir = './data/WT-Planetary Gearbox/40Hz'
    elif args.D_num == 'D4':
        file_dir = './data/WT-Planetary Gearbox/50Hz'
    for i, filename in enumerate(tqdm(os.listdir(file_dir))):
        file_path = os.path.join(file_dir, filename)
        print(file_path)          # ./data/WT-Planetary Gearbox/20Hz\1N1_20.MAT
        data, lab = data_load(args, file_path, label=int(i))
        data1 += data             # <class 'list'> 500 * 5 -> <class 'numpy.ndarray'> (1024,) float32
        lab1 += lab               # <class 'list'> 500 * 5 -> <class 'int'>

    save_dir = './data/save_dataset'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    list_data = [data1, lab1]      # <class 'list'> 2 -> <class 'list'> 500 * 5
    print('*****The data is {} dataset*****'.format(args.D_num), '\n')
    np.save(save_dir + '/' + save_name + '.npy', list_data)

class dataset(Dataset):
    def __init__(self, X, y):
        self.Data = X
        self.Label = y

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label

    def __len__(self):
        return len(self.Data)

class WT_Planetary_Gearbox(object):
    num_sensor = 1
    num_classes = 5
    sampling_rate = 48000
    classes = ["NS",
               "BT", "WG",
               "RC", "MT"]
    # load dataset for operation
    def data_prepare(self, args, save_name):
        data_dir = './data/save_dataset'
        list_data = np.load(data_dir + '/' + save_name + '.npy', allow_pickle=True)
        X_data, y_data = np.vstack(list_data[0]).astype(np.float32), np.array(list_data[1]).astype(np.int32)

        num_classes = 5
        X_data_class = int(len(X_data) / num_classes)
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        for i in range(num_classes):
            X_train.append(X_data[int(X_data_class*i):int(X_data_class*(i+1))][0:int(X_data_class*0.6)])
            y_train.append(y_data[int(X_data_class*i):int(X_data_class*(i+1))][0:int(X_data_class*0.6)])
            X_val.append(X_data[int(X_data_class*i):int(X_data_class*(i+1))][int(X_data_class*0.6):int(X_data_class*0.8)])
            y_val.append(y_data[int(X_data_class*i):int(X_data_class*(i+1))][int(X_data_class*0.6):int(X_data_class*0.8)])
            X_test.append(X_data[int(X_data_class*i):int(X_data_class*(i+1))][int(X_data_class*0.8):])
            y_test.append(y_data[int(X_data_class*i):int(X_data_class*(i+1))][int(X_data_class*0.8):])

        X_train, y_train = np.array(X_train, dtype=np.float32).reshape(-1, args.sample_length), np.array(y_train).reshape(-1)
        X_val, y_val = np.array(X_val, dtype=np.float32).reshape(-1, args.sample_length), np.array(y_val).reshape(-1)
        X_test, y_test = np.array(X_test, dtype=np.float32).reshape(-1, args.sample_length), np.array(y_test).reshape(-1)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(X_train)
        x_val = scaler.transform(X_val)
        x_test = scaler.transform(X_test)

        x_train = np.expand_dims(x_train, axis=1)
        x_val = np.expand_dims(x_val, axis=1)
        x_test = np.expand_dims(x_test, axis=1)

        train_dataset = dataset(x_train, y_train)
        val_dataset = dataset(x_val, y_val)
        test_dataset = dataset(x_test, y_test)
        return train_dataset, val_dataset, test_dataset