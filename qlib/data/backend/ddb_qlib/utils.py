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