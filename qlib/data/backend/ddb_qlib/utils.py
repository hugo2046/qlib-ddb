'''
Author: Hugo
Date: 2025-02-20 23:52:37
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-10-21 22:18:51
Description: 
'''
import re

import pandas as pd
from typing import Union,Tuple,List,Dict

def convert_wind_date_to_datetime(df: pd.DataFrame, type_dict: Union[Dict[str, str], List[Tuple[str, str]]], inplace: bool = False) -> pd.DataFrame:
    """
    将Wind格式的日期转换为datetime格式。

    :param df: 包含Wind格式日期的DataFrame
    :type df: pd.DataFrame
    :param type_dict: 列名和类型的字典或列表
    :type type_dict: Union[Dict[str, str], List[Tuple[str, str]]]
    :param inplace: 是否在原DataFrame上进行修改
    :type inplace: bool
    :return: 转换后的DataFrame
    :rtype: pd.DataFrame

    :说明:
        - 如果inplace为False，将创建DataFrame的副本进行修改
        - 如果type_dict是列表，每个元素应为(列名, 类型)的元组
    """
    if not inplace:
        df = df.copy()

    if isinstance(type_dict, dict):
        items = type_dict.items()
    elif isinstance(type_dict, list):
        items = type_dict
    else:
        raise ValueError("type_dict必须是字典或列表")

    for col, typestr in items:
        if typestr == "DATE":
            df[col] = pd.to_datetime(df[col], format="%Y%m%d")

    return df



def add_dollar_sign_to_params(func_call: str) -> str:
    """
    将函数调用中的参数添加$符号前缀

    参数:
    func_call (str): 函数调用字符串，如 "gtjaAlpha7(close, vol, vwap)"

    返回:
    str: 添加$符号后的函数调用字符串，如 "gtjaAlpha7($close, $vol, $vwap)"
    """
    # 检查输入是否有效
    if not func_call or "(" not in func_call or ")" not in func_call:
        raise ValueError("无效的函数调用格式")

    # 分离函数名和参数部分
    func_name = func_call[: func_call.find("(")]
    params_part = func_call[func_call.find("(") + 1 : func_call.rfind(")")]

    # 如果参数为空，直接返回
    if not params_part.strip():
        return func_call

    # 分割参数并添加$符号
    params = [param.strip() for param in params_part.split(",")]
    dollar_params = [f"${param}" for param in params]

    # 重新组合函数调用
    return f"{func_name}({', '.join(dollar_params)})"


def extract_function_name(func_call: str) -> str:
    """
    从函数调用字符串中提取函数名称，忽略参数部分

    参数:
    func_call (str): 函数调用字符串，如 "gtjaAlpha1(open, close, volume)"

    返回:
    str: 函数名称，如 "gtjaAlpha1"
    """
    # 检查输入是否有效
    if not func_call or "(" not in func_call:
        raise ValueError("无效的函数调用格式")

    # 分离函数名
    func_name = func_call[: func_call.find("(")]
    return func_name

def validate_date_str(date_str: str) -> str:
    """校验 Wind 风格日期字符串（YYYYMMDD），用于拼入 SQL 前的防注入检查。

    :param date_str: 形如 ``"20240101"`` 的日期字符串
    :return: 原样返回通过校验的字符串
    :raises ValueError: 格式不为 8 位数字时抛出
    """
    if not re.fullmatch(r"\d{8}", str(date_str)):
        raise ValueError(f"日期参数必须为 8 位数字(YYYYMMDD)，收到: {date_str!r}")
    return str(date_str)


def validate_sql_identifier(value: str) -> str:
    """校验将拼入 SQL 的标识符/代码类参数（Wind 代码、交易所、库表名等）。

    仅允许字母、数字、点、下划线、连字符——足以覆盖合法的 Wind 代码
    （如 ``000300.SH``）与库表名，同时拒绝引号、分号等注入载体。

    :param value: 待校验的字符串
    :return: 原样返回通过校验的字符串
    :raises ValueError: 含白名单以外字符时抛出
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(value)):
        raise ValueError(f"参数含非法字符(仅允许字母/数字/./_/-): {value!r}")
    return str(value)
