#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
对每个输入903个数据点的窗口样本进行检查、划分、正则化
"""

import numpy as np

DEFAULT_WINDOW = 180

# 检查输入时间序列窗口是否为 903
def is_standard_time_series(time_series, window=DEFAULT_WINDOW):
    """
    Check the length of time_series. If window = 180, then the length of time_series should be 903.
    The mean value of last window should be larger than 0.

    :param time_series: the time series to check, like [data_c, data_b, data_a]
    :type time_series: pandas.Series
    :param window: the length of window
    :return: True or False
    :return type: boolean
    """
    return bool(len(time_series) == 5 * window + 3 and np.mean(time_series[(4 * window + 2):]) > 0)

# 划分 903 的时间数据为 5 部分，[[data_c_left], [data_c_right], [data_b_left], [data_b_right], [data_a]]
def split_time_series(time_series, window=DEFAULT_WINDOW):
    """
    Spilt the time_series into five parts. Each has a length of window + 1

    :param time_series: [data_c, data_b, data_a]
    :param window: the length of window
    :return: spilt list [[data_c_left], [data_c_right], [data_b_left], [data_b_right], [data_a]]
    """
    data_c_left = time_series[0:(window + 1)]
    data_c_right = time_series[window:(2 * window + 1)]
    data_b_left = time_series[(2 * window + 1):(3 * window + 2)]
    data_b_right = time_series[(3 * window + 1):(4 * window + 2)]
    data_a = time_series[(4 * window + 2):]
    split_time_series = [
        data_c_left,
        data_c_right,
        data_b_left,
        data_b_right,
        data_a
    ]
    return split_time_series

# 正则化 划分好的五部分时间序列数据
def normalize_time_series(split_time_series):
    """
    Normalize the split_time_series.

    :param split_time_series: [[data_c_left], [data_c_right], [data_b_left], [data_b_right], [data_a]]
    :return: all list / mean(split_time_series)
    """
    value = np.mean(split_time_series[4])
    if value > 1:
        normalized_data_c_left = list(split_time_series[0] / value)
        normalized_data_c_right = list(split_time_series[1] / value)
        normalized_data_b_left = list(split_time_series[2] / value)
        normalized_data_b_right = list(split_time_series[3] / value)
        normalized_data_a = list(split_time_series[4] / value)
    else:
        normalized_data_c_left = split_time_series[0]
        normalized_data_c_right = split_time_series[1]
        normalized_data_b_left = split_time_series[2]
        normalized_data_b_right = split_time_series[3]
        normalized_data_a = split_time_series[4]
    normalized_split_time_series = [
        normalized_data_c_left,
        normalized_data_c_right,
        normalized_data_b_left,
        normalized_data_b_right,
        normalized_data_a
    ]
    return normalized_split_time_series

# 正则化 划分好的五部分时间序列数据，用最大最小缩放
def normalize_time_series_by_max_min(split_time_series):
    """
    Normalize the split_time_series by max_min_normalization.

    :param split_time_series: [[data_c_left], [data_c_right], [data_b_left], [data_b_right], [data_a]]
    :return: max_min_normalized time_series
    """
    time_series = split_time_series[0] + split_time_series[1][1:] + split_time_series[2] + split_time_series[3][1:] + split_time_series[4]
    max_value = np.max(time_series)
    min_value = np.min(time_series)
    normalized_time_series = [0.0]*len(time_series)
    if max_value - min_value > 0:
        normalized_time_series = list((np.array(time_series) - min_value) / float(max_value - min_value))

    return normalized_time_series
