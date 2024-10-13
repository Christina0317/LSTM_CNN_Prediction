import importlib


def get_class(module_name, class_name='Model'):
    module_model = importlib.import_module(module_name)
    class_model = getattr(module_model,class_name)
    return class_model


def get_trading_dates(start_date, end_date):
    """读取交易日期"""
    with open('/Users/hjx/Documents/projects/Att_CNN_LSTM/data/TradingDates.txt', 'r') as f:
        dates = f.read().splitlines()
    dates = [date for date in dates if (int(date) >= int(start_date)) * (int(date) <= int(end_date))]
    return dates