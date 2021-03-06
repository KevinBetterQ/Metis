#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
定义错误字典
"""

TSD_OP_SUCCESS = 0
TSD_THROW_EXP = 1000
TSD_CHECK_PARAM_FAILED = 1002
TSD_FILE_FORMAT_ERR = 1003
TSD_CAL_FEATURE_ERR = 2001
TSD_READ_FEATURE_FAILED = 2002
TSD_TRAIN_ERR = 2003
TSD_LACK_SAMPLE = 2004

ERR_CODE = {
    TSD_OP_SUCCESS: "操作成功",
    TSD_THROW_EXP: "抛出异常",
    TSD_CHECK_PARAM_FAILED: "参数检查失败",
    TSD_FILE_FORMAT_ERR: "文件格式有误",
    TSD_CAL_FEATURE_ERR: "特征计算出错",
    TSD_READ_FEATURE_FAILED: "读取特征数据失败",
    TSD_TRAIN_ERR: "训练出错",
    TSD_LACK_SAMPLE: "缺少正样本或负样本"
}