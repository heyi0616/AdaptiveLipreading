import os
import glob
import torch
import random
# import librosa
import numpy as np
import sys
from lipreading.utils import read_txt_lines


class MyDataset(object):
    def __init__(self, modality, data_partition, data_dir, label_fp, annonation_direc=None,
                 preprocessing_func=None, data_suffix='.npz', use_boundary=False):
        assert os.path.isfile(label_fp), \
            f"File path provided for the labels does not exist. Path iput: {label_fp}."
        self._data_partition = data_partition
        self._data_dir = data_dir

        self._label_fp = label_fp
        self._annonation_direc = annonation_direc

        self.fps = 25
        self.is_var_length = False
        self.use_boundary = use_boundary
        self.label_idx = -3

        self.preprocessing_func = preprocessing_func

        if self.use_boundary or (self.is_var_length and data_partition == "train"):
            assert self._annonation_direc is not None, \
                "Directory path provided for the sequence timestamp (--annonation-direc) should not be empty."
            assert os.path.isdir(self._annonation_direc), \
                f"Directory path provided for the sequence timestamp (--annonation-direc) does not exist. Directory input: {self._annonation_direc}"

        self.list = self.load_dataset()

    def load_dataset(self):
        # -- read the labels file
        labels = read_txt_lines(self._label_fp)
        file_list = []
        # download from https://github.com/ms-dot-k/User-dependent-Padding/tree/main/data
        split_file = "/home/yihe/repos/User-dependent-Padding-main/data/"
        with open(split_file + 'LRW_ID_{}.txt'.format(self._data_partition), 'r') as f:
            lines = f.readlines()
        for l in lines:
            subject, f_name = l.strip().split()
            file_list.append(os.path.join(self._data_dir, f_name + '.npz'))

        file_list = [f for f in file_list if f.split('/')[self.label_idx] in labels]
        data_list = [[f, labels.index(self._get_label_from_path(f))] for f in file_list]

        print(f"Partition {self._data_partition} loaded")
        return data_list

    def _get_instance_id_from_path(self, x):
        # for now this works for npz/npys, might break for image folders
        instance_id = x.split('/')[-1]
        return os.path.splitext(instance_id)[0]

    def _get_label_from_path(self, x):
        return x.split('/')[self.label_idx]

    def load_data(self, filename):
        assert filename.endswith('npz')
        return np.load(filename)['data']

    def _apply_variable_length_aug(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        info_txt = os.path.join(self._annonation_direc, *filename.split('/')[self.label_idx:])  # swap base folder
        info_txt = os.path.splitext(info_txt)[0] + '.txt'  # swap extension
        info = read_txt_lines(info_txt)

        utterance_duration = float(info[4].split(' ')[1])
        half_interval = int(utterance_duration / 2.0 * self.fps)  # num frames of utterance / 2

        n_frames = raw_data.shape[0]
        mid_idx = (n_frames - 1) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = random.randint(0, max(0, mid_idx - half_interval - 1))  # random.randint(a,b) chooses in [a,b]
        right_idx = random.randint(min(mid_idx + half_interval + 1, n_frames), n_frames)

        return raw_data[left_idx:right_idx]

    def _get_boundary(self, filename, raw_data):
        # read info txt file (to see duration of word, to be used to do temporal cropping)
        info_txt = os.path.join(self._annonation_direc, *filename.split('/')[self.label_idx:])  # swap base folder
        info_txt = os.path.splitext(info_txt)[0] + '.txt'  # swap extension
        info = read_txt_lines(info_txt)

        utterance_duration = float(info[4].split(' ')[1])
        # boundary is used for the features at the top of ResNet, which as a frame rate of 25fps.
        if self.fps == 25:
            half_interval = int(utterance_duration / 2.0 * self.fps)
            n_frames = raw_data.shape[0]
        elif self.fps == 16000:
            half_interval = int(utterance_duration / 2.0 * 25)
            n_frames = raw_data.shape[0] // 640

        mid_idx = (n_frames - 1) // 2  # video has n frames, mid point is (n-1)//2 as count starts with 0
        left_idx = max(0, mid_idx - half_interval - 1)
        right_idx = min(mid_idx + half_interval + 1, n_frames)

        boundary = np.zeros(n_frames)
        boundary[left_idx:right_idx] = 1
        return boundary

    def __getitem__(self, idx):

        raw_data = self.load_data(self.list[idx][0])
        # -- perform variable length on training set
        if (self._data_partition == 'train') and self.is_var_length and not self.use_boundary:
            data = self._apply_variable_length_aug(self.list[idx][0], raw_data)
        else:
            data = raw_data
        preprocess_data = self.preprocessing_func(data)
        label = self.list[idx][1]
        if self.use_boundary:
            boundary = self._get_boundary(self.list[idx][0], raw_data)
            return preprocess_data, label, boundary
        else:
            return preprocess_data, label

    def __len__(self):
        return len(self.list)

