import torch
from models.Moduls import Modul
from loguru import logger
import random
import numpy as np


class ModelRunner:
    def __init__(self, dataset_train, dataset_pred, dataset_params, configs, seed):
        self.dataset_train = dataset_train
        self.dataset_pred = dataset_pred
        self.dataset_params = dataset_params
        self.configs = configs
        self.seed = seed
        self.check_with_model_params()
        self.Models = [Modul(config['model_params']) for config in configs]

    def check_with_model_params(self):
        phase_data_dim_0 = 240 * self.dataset_params['num_train_need_dates'] - (self.dataset_params['embedding_dim'] - 1) * self.dataset_params['tau'] - 1
        phase_data_dim_1 = self.dataset_params['embedding_dim']
        for config in self.configs:
            if config['model_params']['cnn_steps'] != phase_data_dim_0:
                logger.info(f"Config: {config['model_params']['config_name']} about cnn_steps is not matching")
                config['model_params']['cnn_steps'] = phase_data_dim_0
            if config['model_params']['input_dim'] != phase_data_dim_1:
                logger.info(f"Config: {config['model_params']['config_name']} about input_dim is not matching")
                config['model_params']['input_dim'] = phase_data_dim_1
            if config['model_params']['num_steps'] != 240 * self.dataset_params['num_train_need_dates']:
                logger.info(f"Config: {config['model_params']['config_name']} about num_steps is not matching")
                config['model_params']['num_steps'] = 240 * self.dataset_params['num_train_need_dates']

    def Train(self, last_train_epoch, train_end_ic=0.22):
        logger.info('Combine all data starts')
        self.dataset_train.combine_all_data()
        logger.info('Combine all data ends')
        logger.info('Generate batch starts')
        self.dataset_train.generate_batch()
        logger.info('Generate batch ends')

        Models = self.Models
        dataset_train = self.dataset_train
        for Model in Models:
            Model.model = Model.model.cpu()

        epoch = last_train_epoch + 1
        while len(Models) > 0 and epoch < 100:
            cur_epoch_seed = self.seed + epoch
            random.seed(cur_epoch_seed)
            np.random.seed(cur_epoch_seed)

            cur_epoch = str(epoch).zfill(2)
            logger.info(f'Train epoch {cur_epoch} start: ')

            for X_batch, y_batch in dataset_train:
                cur_batch_seed = random.random()
                for Model in Models:
                    torch.manual_seed(cur_batch_seed)
                    Model.Train_batch(X_batch, y_batch)

            for Model in Models:
                Model.Train_epoch_finish(epoch)

            Models = [Model for Model in Models if Model.cur_ic_train < train_end_ic]
            logger.info(f'Train epoch {cur_epoch} finish, {len(Models)} models need to train')
            epoch += 1

    def Predict(self):
        Models = self.Models
        dataset_pred = self.dataset_pred
        for Model in Models:
            Model.model = Model.model.cpu()

        for data, label, key, date in dataset_pred:
            logger.info(f'Predict {date} start: ')
            for Model in Models:
                Model.Predict_date(data, label, key, date)
            logger.info(f'Predict {date} end')

        for Model in Models:
            Model.Predict_finish()