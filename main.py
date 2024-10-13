from loguru import logger
import numpy as np
import pandas as pd
import argparse
import yaml
import os
from models.model_runner import ModelRunner
from utils.base_utils import get_class, get_trading_dates


if __name__ == "__main__":
    is_train = 1
    is_pred = 1
    logger.add("/Users/hjx/Documents/projects/Att_CNN_LSTM/logs/log_20240916.log")

    parser = argparse.ArgumentParser()
    parser.add_argument('-bp', '--base_config_path',
                        default='tau_20_embedding_20_minmax')

    parser.add_argument('-cp', '--configs_path', default='search_params')

    cp = parser.parse_args().configs_path
    bp = parser.parse_args().base_config_path
    base_config_path = f'./configs/base/{bp}.yaml'
    configs_path = f'./configs/{cp}'

    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    base_params = base_config['base_params']
    dataset_params = base_config['dataset_params']
    seed = base_params['seed']

    output_path = os.path.join(base_params['output_path'], bp + '_' + cp)
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"output_path: {output_path}")

    configs = []
    tmp = os.listdir(configs_path)

    for i in tmp:
        config_name = i[:-5]
        logger.info(f"config_name: {config_name}")
        with open(os.path.join(configs_path, i)) as f:
            config = yaml.safe_load(f)
        config['model_params']['output_path'] = output_path
        config['model_params']['config_name'] = config_name
        configs.append(config)

    start_train = base_params['start_train']
    end_train = base_params['end_train']
    dates_train = get_trading_dates(start_train, end_train)
    start_pred = base_params['start_pred']
    end_pred = base_params['end_pred']
    dates_pred = get_trading_dates(start_pred, end_pred)

    dataset_train = get_class('models.dataset_train', 'Dataset')(dates_train, dataset_params, configs)
    dataset_pred = get_class('models.dataset_pred', 'Dataset')(dates_pred, dataset_params)

    model_runner = ModelRunner(dataset_train, dataset_pred, dataset_params, configs, seed)

    if is_train:
        logger.info(f'Train start:')
        last_train_epoch = 0
        model_runner.Train(last_train_epoch)
        logger.info(f'Train end')

    if is_pred:
        logger.info(f'Prediction start:')
        model_runner.Predict()
        logger.info(f'Prediction end')


