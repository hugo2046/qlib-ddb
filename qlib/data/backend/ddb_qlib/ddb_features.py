"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-28 14:00:40
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-28 14:08:23
Description:
"""

import re
from pathlib import Path
from typing import Dict, List, Union, Tuple

import dolphindb as ddb
import pandas as pd


from ....utils import normalize_cache_fields
from ....log import get_module_logger


# qlib表达式与ddb表达式对应表
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


def fetch_features_from_ddb(
    session, instruments, expr, start_time=None, end_time=None, freq="day"
):
    """
    从 DolphinDB 获取特征数据

    Parameters
    ----------
    session : DolphinDB session
        DolphinDB 会话对象
    instruments : Union[str, List[str]]
        证券代码或证券代码列表
    expr : Union[str, List[str]]
        Qlib 格式的表达式或表达式列表
    start_time : str, optional
        开始时间
    end_time : str, optional
        结束时间
    freq : str, default="day"
        数据频率，可选 "day" 或 "min"

    Returns
    -------
    pd.DataFrame
        包含查询结果的数据框，以 ["instrument", "datetime"] 为索引
    """
    from .schemas import QlibTableSchema

    if not isinstance(expr, list):
        expr = [expr]

    _freq: str = "daily" if freq == "day" else "min"
    feature_schema: QlibTableSchema = getattr(QlibTableSchema(), f"feature_{_freq}")()

    db_name: str = feature_schema.db_name
    table_name: str = feature_schema.table_name
    # transform_expression_with_prefix(expr)

    normalized_expr: List[str] = [
        adapt_qlib_expr_syntax_for_ddb(column, OPERATOR_MAPPING, True) for column in expr
    ]

    # 获取基础数据
    base_fields: List[str] = extract_fields_from_expressions(expr)

    # 获取实际查询范围
    start_time, end_time = get_query_date_range(session, start_time, end_time)

    fmt: str = "%Y.%m.%d" if freq == "day" else "%Y.%m.%d %H:%M:%S"
    start_time: pd.Timestamp = pd.to_datetime(start_time).strftime(fmt)
    end_time: pd.Timestamp = pd.to_datetime(end_time).strftime(fmt)

    # 上传时间变量
    upload_dates: str = f"""
    start_time={start_time};
    end_time={end_time};
    """
    session.run(upload_dates)

    # 转换为列表格式
    if isinstance(instruments, str):
        instruments = [instruments]

    # 上传变量
    session.upload(
        {
            "instruments": instruments,
            "expressions": normalized_expr,
            "baseFields": base_fields,
        }
    )

    if is_pure_fields_expressions(expr):
        data: pd.DataFrame = (
            session.loadTable(table_name, f"dfs://{db_name}")
            .select(["code","date"]+base_fields)
            .where(f"date between pair({start_time},{end_time})")
            .where(f"code in {instruments}")
            .sort(["date", "code"])
        ).toDF()

    else:
        ddb_expr = f"""
        ds = CreateDSByDate("{db_name}","{table_name}",start_time,end_time,`date);
        mr(ds, calcPanelFeatures{{,expressions,baseFields,instruments,start_time,end_time}}, , unionAll{{, false}});
        """

        data: pd.DataFrame = session.run(ddb_expr)

    if not isinstance(data, pd.DataFrame):
        raise ValueError("查询结果不是 DataFrame 格式")

    # 格式化结果
    if not data.empty:
        try:
            data: pd.DataFrame = data.set_index(["code", "date"])
        except KeyError:
            raise KeyError("查询结果缺少 'code' 或 'date' 列")

        data.index.names = ["instrument", "datetime"]

    return data.sort_index()


# 重命名ddb_compute_features为fetch_features_from_ddb
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
    freq: str = "day",
) -> pd.DataFrame:
    """
    计算DolphinDB中的特征。

    此函数从DolphinDB数据库中检索和计算特定时间范围内的特定特征。

    :param session: DolphinDB会话对象
    :type session: DDB会话对象

    :param instruments: 代码列表或单个代码字符串（例如股票代码）
    :type instruments: Union[List, str]

    :param fields: 要获取的字段列表或单个字段字符串(Qlib原生表达式)
    :type fields: Union[List, str]

    :param start_time: 数据开始时间
    :type start_time: str

    :param end_time: 数据结束时间
    :type end_time: str

    :param db_name: DolphinDB数据库名称
    :type db_name: str

    :param table_name: DolphinDB表名
    :type table_name: str

    :param mr_by_code: 是否按代码进行MapReduce操作，默认为False。注意：社区版DolphinDB在mr_by_code=True时可能出现OOM问题
    :type mr_by_code: bool, optional

    :param column_name: 时间列的名称，默认为"date"
    :type column_name: str, optional

    :param freq: 数据频率，可选"day"或其他值（如"min"），默认为"day"
    :type freq: str, optional

    :returns: 包含计算特征的DataFrame，按代码和日期排序
    :rtype: pd.DataFrame
    """
    if isinstance(fields, str):
        fields = [fields]

    if isinstance(instruments, str):
        instruments = [instruments]

    fmt: str = "%Y.%m.%d" if freq == "day" else "%Y.%m.%d %H:%M:%S"
    start_time: pd.Timestamp = pd.to_datetime(start_time).strftime(fmt)
    end_time: pd.Timestamp = pd.to_datetime(end_time).strftime(fmt)

    # 标准化qlib表达
    exprs: List[str] = normalize_cache_fields([str(column) for column in fields])
    # escape_backslash=True, 后续用于parseExpr解析
    # 将qlib表达式转换为DolphinDB表达式
    normalized_expr: List[str] = [
        adapt_qlib_expr_syntax_for_ddb(column, OPERATOR_MAPPING, True) for column in exprs
    ]

    # 使用mr_by_code会快很多,但是社区版ddb会出现OOM出现,所有尽量使用mr_by_code=false
    return session.run(
        f"""FeatureEngineering({instruments},dict({normalized_expr},{exprs}),{start_time},{end_time},"{db_name}","{table_name}",{int(mr_by_code)}$bool,"{column_name}")"""
    ).sort_values(["code", "date"])


def get_query_date_range(
    session: ddb.Session, start_dt, end_dt
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """获取查询的日期范围
    根据提供的开始和结束日期，从日历表中确定实际的查询日期范围。
    如果未指定开始或结束日期，则使用日历表中的最早或最晚交易日。
    Parameters
    ----------
    session : ddb.Session
        DolphinDB会话对象
    start_dt : Union[str, pd.Timestamp, None]
        开始日期，如果为None则使用日历表中的最早日期
    end_dt : Union[str, pd.Timestamp, None]
        结束日期，如果为None则使用日历表中的最晚日期
    Returns
    -------
    Tuple[pd.Timestamp, pd.Timestamp]
        处理后的开始日期和结束日期，转换为pandas Timestamp格式
    """

    from .schemas import QlibTableSchema

    db_path: str = f"dfs://{QlibTableSchema.calendar().db_name}"
    tb_path: str = QlibTableSchema.calendar().table_name

    tb: pd.DataFrame = session.loadTable(tb_path, db_path).select("TRADE_DAYS").toDF()

    if start_dt == None:
        start_dt = tb["TRADE_DAYS"].min()

    if end_dt == None:
        end_dt = tb["TRADE_DAYS"].max()

    return pd.to_datetime(start_dt), pd.to_datetime(end_dt)


def adapt_qlib_expr_syntax_for_ddb(
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
                        processed_params = adapt_qlib_expr_syntax_for_ddb(
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


def extract_fields_from_expressions(expressions, rename_map=None):
    """
    从多个表达式中提取所有基础字段变量

    Parameters
    ----------
    expressions : str or list
        表达式或表达式列表，如
        "gtjaAlpha1($open, $close, $vol);" 或
        ["$close/$open", "SMA($high, 10)/$low"]

    rename_map : dict, optional
        字段重命名映射，如 {'vol': 'volume', 'close': 'price_close'}

    Returns
    -------
    list
        所有表达式中提取的去重字段名列表
    """

    # 确保处理列表和单个字符串的表达式
    if not isinstance(expressions, (list, tuple)):
        expressions = [expressions]

    # 正则表达式匹配以$开头的变量名
    pattern = r"\$([a-zA-Z_][a-zA-Z0-9_]*)"

    # 存储所有找到的字段
    all_fields = set()

    # 处理每个表达式
    for expr in expressions:
        # 如果是另一个嵌套列表，递归处理
        if isinstance(expr, (list, tuple)):
            nested_fields = extract_fields_from_expressions(expr, None)
            all_fields.update(nested_fields)
        else:
            # 提取当前表达式中的字段
            fields = re.findall(pattern, expr)
            all_fields.update(fields)

    # 转换成列表并排序，保证结果稳定
    result = sorted(list(all_fields))

    # 应用重命名映射（如果提供）
    if rename_map:
        result = [rename_map.get(field, field) for field in result]

    return result


def is_pure_fields_expressions(exprs):
    """
    检查表达式列表是否只包含纯字段引用（如$close, $open）

    Parameters:
    -----------
    exprs: List[str]
        表达式列表

    Returns:
    --------
    bool
        如果所有表达式都是纯字段引用则返回True，否则返回False
    """

    if not isinstance(exprs, (list, tuple)):
        exprs = [exprs]

    pattern = r"^\$([a-zA-Z_][a-zA-Z0-9_]*)$"

    for expr in exprs:
        # 检查是否为纯字段引用格式
        if not re.match(pattern, expr):
            return False

    return True
