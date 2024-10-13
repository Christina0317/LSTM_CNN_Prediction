import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
from models.normalization import *
import pickle


class Dataset(object):
    def __init__(self, dates, dataset_params):
        self.data_path = dataset_params['pred_data_path']
        self.dates = dates
        self.dataset_params = dataset_params
        self.tau = dataset_params['tau']
        self.embedding_dim = dataset_params['embedding_dim']
        self.phase_data_dim_0 = 240 * dataset_params['num_train_need_dates'] - (self.embedding_dim - 1) * self.tau - 1
        self.phase_data_dim_1 = self.embedding_dim

        self.normalize = eval(self.dataset_params['norm_method'])

    def __iter__(self):
        for date in self.dates:
            yield self.parse(date)

    def clip(self, arr, clip_value):
        arr[arr > clip_value] = clip_value
        arr[arr < -clip_value] = -clip_value
        return arr

    def parse(self, date):
        ori_x_path = self.data_path + '/x_' + date + '.pkl'
        ori_y_path = self.data_path + '/y_' + date + '.pkl'
        with open(ori_x_path, 'rb') as f:
            x = pickle.load(f)
        with open(ori_y_path, 'rb') as f:
            y = pickle.load(f)

        # Generate px
        px = []
        for s in range(len(x)):
            timeseries = x.iloc[s]
            px_s = np.zeros((self.phase_data_dim_0, self.embedding_dim))
            for i in range(self.phase_data_dim_0):
                px_s[i, :] = [timeseries[i + j * self.tau] for j in range(self.embedding_dim)]
            px.append(px_s)

        keys = list(x.index)
        x = np.array(x.values)
        label = np.array(y.values)
        px = np.array(px)

        if self.dataset_params['ts_norm']:
            x = self.normalize(x)
            px = self.normalize(px)
        data = (x, px)

        return data, label[:, 0], keys, date
