def min_max_normalize(data, min_value=0.0, max_value=1.0):
    data_min = min(data)
    data_max = max(data)
    denom = data_max - data_min
    denom[denom == 0] = 1  # 避免除以零

    normalized_data = (data - data_min) / denom
    normalized_data = normalized_data * (max_value - min_value) + min_value
    return normalized_data


def Z_normalization(arr):
    dim = 1
    eps = 1e-5
    arr = (arr-arr.mean(axis=dim,keepdims=True))/(arr.std(axis=dim,keepdims=True)+eps)
    return arr


def Min_max_scaling(arr):
    dim = 1
    eps = 1e-5
    arr = (arr - arr.min(axis=dim,keepdims=True))/(arr.max(axis=dim,keepdims=True)-arr.min(axis=dim,keepdims=True)+eps)
    return arr