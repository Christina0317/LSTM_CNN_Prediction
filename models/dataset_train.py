import os
import numpy as np
import pandas as pd
import torch
import random
from multiprocessing import Process, Manager, Lock
import time
from datetime import datetime
from models.normalization import *
import pickle
from loguru import logger


class Dataset(object):
    def __init__(self, dates, dataset_params, configs):
        self.dataset_params = dataset_params
        self.configs = configs
        self.original_data_path = dataset_params['train_original_data_path']
        self.phase_data_path = dataset_params['train_phase_data_path']
        self.tau = dataset_params['tau']
        self.embedding_dim = dataset_params['embedding_dim']
        self.dates = dates
        self.dataset_params = dataset_params
        self.phase_data_dim_0 = 240 * dataset_params['num_train_need_dates'] - (self.embedding_dim-1) * self.tau - 1
        self.phase_data_dim_1 = self.embedding_dim

        self.batched_list = None
        self.data = None

        self.normalize = eval(self.dataset_params['norm_method'])

        # params for multiprocessing
        self.buffer_threads = None
        self.instances_buffer_size = 50
        self.num_process = 3
        manager = Manager()
        self.buffer_single = manager.dict()
        self.cur_yield_index = manager.Value('i', 0)
        # self.buffer_lock = Lock()

    def combine_all_data(self):
        data_all = {}
        if self.phase_data_path is not None:
            for date in self.dates:
                ori_x_path = self.original_data_path + '/x_' + date + '.pkl'
                ori_y_path = self.original_data_path + '/y_' + date + '.pkl'
                phase_path = self.phase_data_path + '/px_' + date + '.pkl'
                with open(ori_x_path, 'rb') as f:
                    x = pickle.load(f)
                with open(ori_y_path, 'rb') as f:
                    y = pickle.load(f)
                with open(phase_path, 'rb') as f:
                    px = pickle.load(f)
                for i in range(len(px)):
                    data_all[date+'_'+x.index[i]] = x.iloc[i], px.iloc[i], y.iloc[i]
        else:
            for date in self.dates:
                ori_x_path = self.original_data_path + '/x_' + date + '.pkl'
                ori_y_path = self.original_data_path + '/y_' + date + '.pkl'
                with open(ori_x_path, 'rb') as f:
                    x = pickle.load(f)
                with open(ori_y_path, 'rb') as f:
                    y = pickle.load(f)
                # px = []
                for s in range(len(x)):
                    timeseries = x.iloc[s]
                    code = x.index[s]
                    px_s = np.zeros((self.phase_data_dim_0, self.embedding_dim))
                    for i in range(self.phase_data_dim_0):
                        px_s[i, :] = [timeseries[i + j * self.tau] for j in range(self.embedding_dim)]
                    data_all[date + '_' + code] = x.iloc[s], px_s, y.iloc[s]
        self.data = data_all

    def generate_batch(self, shuffle=True):
        key_items = list(self.data.keys())
        if shuffle:
            random.shuffle(key_items)
        self.batched_list = [key_items[i:i + self.dataset_params['batch_size']] for i in
                                   range(0, len(key_items), self.dataset_params['batch_size']) if
                                   len(key_items[i:i + self.dataset_params['batch_size']]) == self.dataset_params[
                                       'batch_size']]

    def clip(self, arr, clip_value):
        arr[arr > clip_value] = clip_value
        arr[arr < -clip_value] = -clip_value
        return arr

    def __iter__(self):
        for i in range(len(self.batched_list)):
            yield self.parse(i)

    def parse(self, read_index):
        # 获取一个batch的数据
        keys_batched = self.batched_list[read_index]

        x_batch = []
        px_batch = []
        y_batch = []
        for i in range(len(keys_batched)):
            key_item = keys_batched[i]

            x, px, y = self.data[key_item]
            px = px[:self.phase_data_dim_0, :self.phase_data_dim_1]

            x_batch.append(x)
            px_batch.append(px)
            y_batch.append(y)

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        px_batch = np.array(px_batch)

        if self.dataset_params['ts_norm']:
            x_batch = self.normalize(x_batch)
            px_batch = self.normalize(px_batch)

        X_batch = (x_batch, px_batch)
        return X_batch, y_batch[:, 0]

