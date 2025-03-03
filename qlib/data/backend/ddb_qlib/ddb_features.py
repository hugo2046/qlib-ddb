"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-28 14:00:40
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-28 14:08:23
Description: 
"""

import re
from pathlib import Path
from typing import Dict, List, Union

import dolphindb as ddb
import pandas as pd


from ....utils import normalize_cache_fields
from ....log import get_module_logger

OPERATOR_MAPPING: Dict = {
    "Abs": "abs",
    "Sign": "signum",
    "Log": "log",
    "Mask": "mask",
    "Not": "not",
    "Power": "power",
    "Add": "add",
    "Sub": "sub",
    "Mul": "mul",
    "Div": "div",
    "Greater": "max",
    "Less": "min",
    "Gt": "gt",
    "Ge": "ge",
    "Lt": "lt",
    "Le": "le",
    "Eq": "eq",
    "Ne": "ne",
    "And": "and",
    "Or": "or",
    "If": "iif",
    "Ref": "move",
    "Mean": "mavg",
    "Sum": "msum",
    "Std": "mstd",
    "Var": "mvar",
    "Skew": "mskew",
    "Kurt": "mkurtosis",
    "Max": "mmax",
    "IdxMax": "mimax",
    "Min": "mmin",
    "IdxMin": "mimin",
    "Med": "mmed",
    "Mad": "mmad",
    "Rank": "rolling_rank",
    "Count": "mcount",
    "Slope": "mslr",
    "Resi": "mmse",
    "WMA": "mwavg",
    "EMA": "ema",
    "Corr": "mcorr",
    "Cov": "mcovar",
    "TResample": "resample",
    "Quantile": "rolling_quantile",
    "Delta": "rolling_delta",
    "Rsquare": "Rsquare",
    "Resi": "Resi",
    "Slope": "Slope",
}


def register_ddb_functions_to_qlib(session: ddb.Session) -> None:
    """
    在 DolphinDB 会话中注册与 qlib 对应的自定义函数

    这些函数实现了 qlib 中的特定操作，使 DolphinDB 能够兼容 qlib 的表达式计算

    参数:
    - session: DolphinDB 会话实例
    """

    # 在会话中执行函数定义脚本

    script_path = Path(__file__).parent / "ddb_scripts"
    for script_file in script_path.glob("*.dos"):
        session.runFile(script_file)
    get_module_logger("ddb_features").info("已注册 qlib 兼容函数到 DolphinDB 会话")



def ddb_compute_features(
    session,
    instruments: Union[List, str],
    fields: Union[List, str],
    start_time: str,
    end_time: str,
    db_name: str,
    table_name: str,
    mr_by_code: bool = False,
    column_name: str = "date",
    freq:str="day"
) -> pd.DataFrame:

    if isinstance(fields, str):
        fields = [fields]

    if isinstance(instruments, str):
        instruments = [instruments]

    fmt:str = "%Y.%m.%d" if freq == "day" else "%Y.%m.%d %H:%M:%S"
    start_time:pd.Timestamp = pd.to_datetime(start_time).strftime(fmt)
    end_time:pd.Timestamp = pd.to_datetime(end_time).strftime(fmt)

    # 标准化qlib表达
    exprs: List[str] = normalize_cache_fields([str(column) for column in fields])
    # escape_backslash=True, 后续用于parseExpr解析
    normalized_expr: List[str] = [
        convert_qlib_to_dolphindb(column, OPERATOR_MAPPING, True) for column in exprs
    ]

    # 使用mr_by_code会快很多,但是社区版ddb会出现OOM出现,所有尽量使用mr_by_code=false
    return session.run(
        f"""FeatureEngineering({instruments},dict({normalized_expr},{exprs}),{start_time},{end_time},"{db_name}","{table_name}",{int(mr_by_code)}$bool,"{column_name}")"""
    ).sort_values(["code", "date"])


def convert_qlib_to_dolphindb(
    expr: str, operator_mapping: Dict = OPERATOR_MAPPING, escape_backslash: bool = False
) -> str:
    """
    将 qlib 表达式转换为 DolphinDB 表达式，支持复杂嵌套结构

    参数:
    - expr: qlib 格式的表达式，如 "EMA($CLOSE,10)/Rank($CLOSE,3)"
    - operator_mapping: 操作符映射字典
    - escape_backslash: 是否转义反斜杠(用于parseExpr解析), 默认False

    返回:
    - 转换后的 DolphinDB 表达式
    """
    # 移除所有 $ 符号
    expr = expr.replace("$", "")

    # 定义括号配对堆栈
    bracket_stack = []
    function_starts = []
    i = 0

    # 查找所有函数调用
    while i < len(expr):
        if expr[i : i + 1].isalpha() or expr[i : i + 1] == "_":
            # 找到一个可能的函数名
            start = i
            while i < len(expr) and (
                expr[i : i + 1].isalnum() or expr[i : i + 1] == "_"
            ):
                i += 1
            func_name = expr[start:i]

            # 跳过空白字符
            while i < len(expr) and expr[i : i + 1].isspace():
                i += 1

            # 检查是否是函数调用 (是否后面跟着左括号)
            if i < len(expr) and expr[i : i + 1] == "(":
                function_starts.append((start, i, func_name))
                bracket_stack.append(i)
                i += 1
            continue

        elif expr[i : i + 1] == "(":
            bracket_stack.append(i)
            i += 1
        elif expr[i : i + 1] == ")":
            if bracket_stack:
                start = bracket_stack.pop()

                # 如果这个右括号匹配的是函数调用的左括号
                for j in range(len(function_starts) - 1, -1, -1):
                    func_start, func_bracket, func_name = function_starts[j]
                    if func_bracket == start:
                        # 得到完整的函数调用
                        # full_func = expr[func_start : i + 1]
                        params = expr[start + 1 : i]

                        # 递归处理参数
                        processed_params = convert_qlib_to_dolphindb(
                            params, operator_mapping
                        )

                        # 替换函数名
                        if func_name in operator_mapping:
                            new_func_name = operator_mapping[func_name]
                        else:
                            new_func_name = func_name

                        # 构建新的函数调用
                        replacement = f"{new_func_name}({processed_params})"

                        # 在表达式中替换这个函数调用
                        expr = expr[:func_start] + replacement + expr[i + 1 :]

                        # 更新索引和状态
                        delta = len(replacement) - (i + 1 - func_start)
                        i += delta
                        function_starts.pop(j)
                        break

            i += 1
        else:
            i += 1

    # 最后处理运算符
    operators_pattern = r"([+\-*/^=<>!&|]+)"
    segments = re.split(operators_pattern, expr)

    # 仅对保留的部分应用操作符映射
    # for i in range(len(segments)):
    #     segment = segments[i].strip()
    #     if segment in operator_mapping:
    #         segments[i] = operator_mapping[segment]
    # return "".join(segments)

    # 处理分割后的内容和操作符
    result_segments = []
    for i in range(len(segments)):
        segment = segments[i].strip()

        # 替换运算符
        # NOTE:给ddb的parseExpr函数解析"\"需要考虑转码"\\"
        if segment == "/":
            # 根据escape_backslash参数决定使用单反斜杠还是双反斜杠
            if escape_backslash:
                result_segments.append("\\")  # 双反斜杠，用于parseExpr
            else:
                result_segments.append(chr(92))  # 单反斜杠，用于SQL查询
        elif segment in operator_mapping:
            result_segments.append(operator_mapping[segment])
        else:
            result_segments.append(segment)

    return "".join(result_segments)
