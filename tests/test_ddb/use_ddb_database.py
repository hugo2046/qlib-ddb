"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-21 10:58:34
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-21 14:23:23
Description: 
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[0].parent)
sys.path.insert(0, PROJECT_ROOT)

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from typing import List
from qlib.data.data import LocalExpressionProvider
from qlib.utils import parse_field

import pandas as pd


def test_instruments():
    # 获取股票池
    pool = D.instruments("ashares")
    codes: List[str] = D.list_instruments(pool, as_list=True)
    print("股票池:")
    print(codes[:10])


def test_calendar():
    # 获取交易日历
    calcendar = D.calendar()
    print("交易日历:")
    print(calcendar[-10:])


def test_features():
    
    from qlib.contrib.data.loader import Alpha158DL
    import time

    start_time = time.time()

    conf = {
        "kbar": {},
        "price": {
            "windows": [0],
            "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
        },
        "rolling": {},
    }

    feature_expr = Alpha158DL().get_feature_config(conf)

    # 获取特征
    pool = D.instruments("ashares")
    codes: List[str] = D.list_instruments(pool, as_list=True)
    print(f"股票池:{len(codes)}")
    print("计算因子:")
    # 405.46 秒
    features: pd.DataFrame = D.features(
        codes,
        start_time="2021-01-01",
        end_time="2024-04-12",
        fields=feature_expr[0][:3],
    )
    print("结果：")
    print(features)

    end_time = time.time()
    print(f"计算耗时: {end_time - start_time:.2f} 秒")


def test_ohlc_features():
    pool = D.instruments("ashares")
    codes: List[str] = D.list_instruments(pool, as_list=True)

    features: pd.DataFrame = D.features(
        codes[:2],
        start_time="2021-01-01",
        end_time="2024-04-12",
        fields=["$close", "$open", "$high", "$low"],
    )
    print("结果：")
    print(features)


def test_storage():
    # 测试storage
    from qlib.data.storage.dolphindb_storage import DBFeatureStorage

    print(DBFeatureStorage("000038.SZ", "S_DQ_ADJCLOSE", "day")[2455:2700])


def test_expression_provider():
    # 测试单因子表达
    from qlib.data.ops import Operators, Feature

    field = "(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close"
    print(eval(parse_field(field))._load_internal("000038.SZ", 2455, 2700, "day"))


def test_use_cfg():

    from qlib.utils import init_instance_by_config

    pool = D.instruments("ashares")

    data_handler_config = {
        "start_time": "2021-01-01",
        "end_time": "2023-12-31",
        "fit_start_time": "2022-01-01",
        "fit_end_time": "2022-02-28",
        "instruments": pool,
    }

    # 任务参数配置
    task = {
        # 机器学习模型参数配置
        "model": {
            # 模型类
            "class": "LGBModel",
            # 模型类所在模块
            "module_path": "qlib.contrib.model.gbdt",
            # 模型类超参数配置，未写的则采用默认值。这些参数传给模型类
            "kwargs": {  # kwargs用于初始化上面的class
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
                "early_stopping_rounds": 50,  # 训练迭代提前停止条件
                "num_boost_round": 1000,  # 最大训练迭代次数
            },
        },
        "dataset": {  # 　因子数据集参数配置
            # 数据集类，是Dataset with Data(H)andler的缩写，即带数据处理器的数据集
            "class": "DatasetH",
            # 数据集类所在模块
            "module_path": "qlib.data.dataset",
            # 数据集类的参数配置
            "kwargs": {
                "handler": {  # 数据集使用的数据处理器配置
                    "class": "Alpha158",  # 数据处理器类，继承自DataHandlerLP
                    "module_path": "qlib.contrib.data.handler",  # 数据处理器类所在模块
                    "kwargs": data_handler_config,  # 数据处理器参数配置
                },
                "segments": {  # 数据集时段划分
                    "train": ("2021-01-01", "2021-12-31"),  # 训练集时段
                    "valid": ("2022-01-01", "2022-02-28"),  # 验证集时段
                    "test": ("2022-03-01", "2023-12-31"),  # 测试集时段
                },
            },
        },
    }

    dataset = init_instance_by_config(task["dataset"])  # 类型DatasetH
    print(dataset)


if __name__ == "__main__":

    use_database = True
    print(f"调用的qlib文件:{qlib.__file__}")

    if use_database:
        uri = "dolphindb://admin:123456@114.80.110.170:28848"
        qlib.init(
            database_uri=uri,
            region=REG_CN,
        )
    else:
        qlib.init(provider_uri="/qlib_data/cn_data", region=REG_CN)

    # test_instruments()
    # test_calendar()
    # test_features()
    # test_storage()
    # test_ohlc_features()
    # test_use_cfg()
    test_expression_provider()
