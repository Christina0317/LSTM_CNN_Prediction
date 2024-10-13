import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from multiprocessing import Pool
from loguru import logger
from tqdm import tqdm
import pickle


def mutual_information(x, tau):
    """计算互信息"""
    def entropy(signal):
        """ 估计信号的熵 """
        kde = gaussian_kde(signal)
        return -np.sum(kde(signal) * np.log(kde(signal)))

    x_t = x[:-tau]
    x_t_tau = x[tau:]
    if np.ndim(x) == 1:
        x_t = x_t.reshape(-1, len(x_t))
        x_t_tau = x_t_tau.reshape(-1, len(x_t_tau))
    joint_h = entropy(np.vstack((x_t, x_t_tau)))
    h_x_t = entropy(x_t)
    h_x_t_tau = entropy(x_t_tau)

    return h_x_t + h_x_t_tau - joint_h


def calculate_tau(x, max_tau, if_plot=False):
    """
    用于确定延迟时间tau
    :param x: 时间序列
    :param max_tau: 限制的最大的延迟时间
    :param plot: 是否画图
    :return: int -> 延迟时间tau
    """
    mi = []
    for tau in range(1, max_tau + 1):
        mi.append(mutual_information(x, tau))
    if if_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, max_tau + 1), mi, marker='o')
        plt.title('mutual information and τ')
        plt.xlabel('τ')
        plt.ylabel('mutual information')
        plt.grid(True)
        plt.show()
    return np.argmin(mi) + 1  # 返回最小互信息值的索引位置


def cao_method(time_series, tau, max_dim=20):
    """
    cao法确定嵌入维数m
    :param time_series:
    :param tau:
    :param max_dim:
    :return:
    """
    def distance(vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def E1(d):
        embeddings = np.array([time_series[i:i + d * tau] for i in range(len(time_series) - d * tau)])
        distances = np.array([distance(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)])
        return np.mean(distances)

    E1_values = [E1(d) for d in range(1, max_dim + 1)]
    E1_ratios = [E1_values[i] / E1_values[i + 1] for i in range(len(E1_values) - 1)]

    for i, ratio in enumerate(E1_ratios):
        if ratio < 0.1:
            return i + 1  # 嵌入维度为 i+1，因为维度从1开始计数

    return max_dim  # 如果没有找到合适的阈值，返回最大维度


def generate_phase_dataset(timeseries, if_plot=False):
    # Reconstruct x
    try:
        tau = calculate_tau(timeseries, 20, if_plot=False)
        emb_dim = cao_method(timeseries, tau)
    except:
        tau = 1
        emb_dim = 20
    M = len(timeseries) - (emb_dim - 1) * tau
    x = np.zeros((M - 1, emb_dim))
    for i in range(M-1):
        x[i, :] = [timeseries[i+j*tau] for j in range(emb_dim)]

    # Reconstruct y
    y = timeseries[1:M]

    if if_plot:
        from sklearn.neighbors import NearestNeighbors
        from mpl_toolkits.mplot3d import Axes3D

        plt.figure(figsize=(10, 8))
        if emb_dim > 2:
            ax = plt.subplot(111, projection='3d')
            ax.plot(x[:, 0], x[:, 1], x[:, 2])
            ax.set_title('3D Phase Space Reconstruction')
        else:
            plt.plot(x[:, 0], x[:, 1])
            plt.title('2D Phase Space Reconstruction')
        plt.show()

    return x, y, tau, emb_dim


def generate_phase_dataset_all(data_all, num_processes=8):
    # with Pool(num_processes) as pool:
    #     results = list(tqdm(pool.imap(generate_phase_dataset, [(timeseries, False) for timeseries in data_all]),
    #                         total=len(data_all), desc="Processing datasets"))
    px_all = {}
    py_all = {}
    for date in data_all.keys():
        data = data_all[date]
        px_date = []
        py_date = []
        for i in tqdm(range(len(data))):
            timeseries = data[i]
            px, py, _, _ = generate_phase_dataset(timeseries, False)
            # px, py, _, _ = results
            px_date.append(px)
            py_date.append(py)
        px_all[date] = px_date
        py_all[date] = py_date
        path_x = '/Users/hjx/Documents/projects/Att_CNN_LSTM/data/phase_data/px_' + date + '.pkl'
        with open(path_x, 'wb') as f:
            pickle.dump(px_date, f)
        path_y = '/Users/hjx/Documents/projects/Att_CNN_LSTM/data/phase_data/py_' + date + '.pkl'
        with open(path_y, 'wb') as f:
            pickle.dump(py_date, f)
        logger.info(f"get phase data: {date}")
    return px_all, py_all


def phase_data_processing(px, py):
    shape_mat = []
    for i in range(len(px)):
        shape = np.shape(px[i])
        shape_mat.append(shape)
    shape_mat = np.array(shape_mat)
    min_step = min(shape_mat[:, 0])
    min_emb = min(shape_mat[:, 1])

    px_new = []
    py_new = []
    for i in range(len(px)):
        x = px[i]
        x = x[:min_step, :min_emb]
        y = py[i]
        y = y[:min_step]
        px_new.append(x)
        py_new.append(y)
    return np.array(px_new), np.array(py_new)


if __name__ == "__main__":
    path = '/Volumes/E/quant_data/db_jq_candle_minute_w_nan.h5'
    data = pd.read_hdf(path, key='20230306')
    df = data[data['code']=='000637.XSHE']
    x = df['close'].values
    # x = x.reshape(-1, len(x))
    restruct_x, restruct_y, tau, emb_dim = generate_phase_dataset(x, if_plot=True)
