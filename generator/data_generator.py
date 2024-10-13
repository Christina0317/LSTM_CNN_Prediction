import pandas as pd
import numpy as np
import math
import random
from loguru import logger
from tqdm import tqdm


random.seed(42)
np.random.seed(42)


class DataGenerator:
    def __init__(self, num_dates, start_date, end_date):
        self.num_dates = num_dates
        self.start_date = start_date
        self.end_date = end_date
        self.path = '/Volumes/E/quant_data/db_jq_candle_minute_w_nan.h5'

        self.trading_codes = self.get_trading_codes()
        self.trading_dates = self.get_trading_dates()

    def get_trading_dates(self):
        """读取交易日期"""
        with open('/Users/hjx/Documents/projects/Att_CNN_LSTM/data/TradingDates.txt', 'r') as f:
            dates = f.read().splitlines()
        dates = [date for date in dates if (int(date) >= int(self.start_date)) * (int(date) <= int(self.end_date))]
        return dates

    def get_trading_codes(self):
        """读取交易代码"""
        with open('/Users/hjx/Documents/projects/Att_CNN_LSTM/data/TradingCodes.txt', 'r') as f:
            code_list = f.read().splitlines()
        return code_list

    def get_days_data(self, date_list):
        """
        :param date_list: 日期列表, 包括训练数据集和预测数据集
        :return: batch_data -> list
        """
        ### 获取数据
        data = {}
        for date in date_list:
            df = pd.read_hdf(self.path, key=date)
            for code in self.trading_codes:
                if code not in data.keys():
                    data[code] = df[df['code'] == code]['close'].values
                else:
                    data[code] = np.append(data[code], df[df['code'] == code]['close'].values)
        batch_data = []
        for code in data.keys():
            if len(data[code]) == (self.num_dates + 1) * 240:
                batch_data.append(data[code])

        ### 对batch_data进行缺失值处理
        # 若最后一列为nan，则删除对应的一整行
        batch_data = [row for row in batch_data if not math.isnan(row[-1])]
        # 对剩下的行进行 NaN 值向后填充
        for row in batch_data:
            last_valid = 0  # 用来保存前一个有效值
            for i in range(len(row)):
                if math.isnan(row[i]):
                    row[i] = last_valid  # 用前一个有效值填充 NaN
                else:
                    last_valid = row[i]  # 更新最后一个有效值为当前值

        return batch_data

    def fill_nan(self, row):
        """向后填充NaN值"""
        mask = np.isnan(row)
        if np.any(mask):
            row[mask] = np.nan
            np.nan_to_num(row, copy=False, nan=np.nanmax(row))  # 填充上一个有效值
        return row

    def get_days_data_1(self, date_list):
        """
        获取指定日期范围内的批次数据。
        :param date_list: 日期列表, 包括训练数据集和预测数据集
        :return: batch_data -> list, len(batch_data)=这几天时间在股票市场存在的股票
        """
        ### 获取数据，使用向量化的方式来处理
        data = {code: [] for code in self.trading_codes}

        for date in date_list:
            df = pd.read_hdf(self.path, key=date)  # 读取当天的数据
            df = df[df['code'].isin(self.trading_codes)]  # 筛选所需的交易代码

            for code in self.trading_codes:
                data[code].append(df[df['code'] == code]['close'].values)

        # 将每只股票的所有日期数据拼接起来
        data = {code: np.concatenate(val) for code, val in data.items() if len(val) == len(date_list)}
        data = {code: val for code, val in data.items() if len(val) == (self.num_dates + 1) * 240 and not np.isnan(val[-1])}

        keys = []
        batch_data = []
        for code, val in data.items():
            keys.append(code)
            batch_data.append(self.fill_nan(val))

        return np.array(batch_data), keys

    def batch_dates(self):
        len_dates = len(self.trading_dates)
        dates_list = [self.trading_dates[i-self.num_dates:(i+1)] for i in range(self.num_dates, len_dates)]
        # 取第i天和之前的几天
        return dates_list

    def get_data(self):
        date_lists = self.batch_dates()
        x_data = {}
        y_data = {}
        # shape(data_all) -> [len(date_lists)*len(code_list), 240*(self.num_dates+1)]
        # 用 i-num_dates:i 去预测第 i 天的涨跌幅
        for i in range(len(date_lists)):
            date_list = date_lists[i]
            data, keys = self.get_days_data_1(date_list)
            logger.info(f"get date list data: {date_list[-2]}")
            x_data[date_list[-2]] = pd.DataFrame(data[:, :self.num_dates*240], index=keys)
            y_data[date_list[-2]] = pd.DataFrame((data[:, -1] - data[:, -240]) / (data[:, -240] + 1e-6) * 100, index=keys)

        return x_data, y_data


if __name__ == '__main__':
    num_dates = 5
    start_date = '20231201'
    end_date = '20231231'
    data_generator = DataGenerator(num_dates, start_date, end_date)

    x_batch_data, y_batch_data = data_generator.get_data()

