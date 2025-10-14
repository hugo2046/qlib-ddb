"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-28 14:00:40
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-28 14:08:23
Description:
"""

import bisect
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import dolphindb as ddb
import numpy as np
import pandas as pd
from packaging import version

from ....log import get_module_logger
from ....utils import normalize_cache_fields
from .schemas import QlibTableSchema

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


##################################################################################################################


def normalize_fields_to_ddb(
    fields: Union[str, List[str], Dict],
) -> Tuple[Dict, List, bool]:
    """
    规范化字段为 DolphinDB 表达式，并保持输入顺序（list 或 dict 的顺序）。
    - 支持 str / list / dict 输入。
    - 对 list 输入利用 dict comprehension 去重且保持顺序（fields_od 的 key 顺序）。
    - 逐项调用 normalize_cache_fields([expr]) 以保证归一化结果按 fields_od 顺序产生。
    - 若归一化后的 normalized key 重复则跳过后续重复项（与 fields_od 的去重语义一致）。
    - 若不同输入在 adapt 为 DolphinDB 表达式后冲突，则抛错提醒。
    """
    # 统一为 list
    if isinstance(fields, str):
        fields = [fields]

    # 保留输入顺序并去重：list -> dict comprehension（保持顺序），dict 保持原有顺序（py3.7+）
    if isinstance(fields, list):
        fields_od = {expr: f"ExprName{i}" for i, expr in enumerate(fields)}
    elif isinstance(fields, dict):
        fields_od = fields
    else:
        raise TypeError("fields must be str, list or dict")

    # 逐项归一化，按 fields_od key 顺序，归一化阶段去重
    exprs: Dict[str, str] = {}
    for expr, alias in fields_od.items():
        nk_list = normalize_cache_fields([expr])
        if not nk_list:
            raise ValueError(f"normalize_cache_fields 返回空结果，输入: {expr}")
        nk = nk_list[0]
        if nk in exprs:
            # 已存在相同 normalized key，跳过（与 fields_od 的去重语义一致）
            continue
        exprs[nk] = alias

    # adapt 为 DolphinDB 表达式，保持顺序；检测 adapt 后的冲突
    normalized_expr: Dict[str, str] = {}
    for k_norm, alias in exprs.items():
        adapted = adapt_qlib_expr_syntax_for_ddb(k_norm, OPERATOR_MAPPING, True)
        if adapted in normalized_expr:
            raise ValueError(f"归一化后出现重复的DolphinDB表达式: {adapted}")
        normalized_expr[adapted] = alias

    # --- 在此内联实现 base_fields 提取并保持输入顺序（不再依赖 extract_fields_from_expressions） ---
    pattern = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")
    base_fields: List[str] = []
    # 过滤掉 code 和 date 字段
    excluded_fields = {"code", "date"}

    for expr in exprs.keys():  # exprs 已按输入顺序构建
        for fld in pattern.findall(expr):
            if fld not in base_fields and fld not in excluded_fields:
                base_fields.append(fld)

    # --- 在此内联实现是否为纯字段表达式的判断（基于原始输入顺序 fields_od） ---
    pure_pattern = re.compile(r"^\$([a-zA-Z_][a-zA-Z0-9_]*)$")
    is_pure_fields: bool = all(bool(pure_pattern.match(e)) for e in fields_od.keys())

    return normalized_expr, base_fields, is_pure_fields


def build_field_expr(
    session: ddb.Session, db_name: str, table_name: str, base_fields: List[str]
) -> List[str]:
    """
    构建字段表达式列表，用于查询表时确保所有基础字段都被包含。

    如果基础字段在表的列中存在，则直接返回字段名；
    如果不存在，则返回 "0 as 字段名" 以保证查询结果中包含该字段，值为0。

    :param session: DolphinDB会话对象，用于加载数据表。
    :type session: Session
    :param db_name: 数据库名称。
    :type db_name: str
    :param table_name: 表名称。
    :type table_name: str
    :param base_fields: 需要查询的基础字段列表。
    :type base_fields: List[str]
    :return: 字段表达式列表，每个元素为字段名或"0 as 字段名"的表达式。
    :rtype: List[str]
    """
    tb = session.loadTable(table_name, f"dfs://{db_name}")
    table_columns: List[str] = tb.schema["name"].tolist()

    return [col if col in table_columns else f"0 as {col}" for col in base_fields]


# ddb_features使用
def fetch_features_from_ddb(
    session: ddb.Session,
    instruments: Union[List[str], Dict],
    fields: Union[List[str], str, Dict],
    start_time: Union[pd.Timestamp, int] = None,
    end_time: Union[pd.Timestamp, int] = None,
    freq: str = "day",
):
    """
    从 DolphinDB 获取特征数据

    与ddb_compute_features的区别在于fetch_features_from_ddb将基础字段close,high,low等转为panel数据，
    即index-date,columns-code,values这种形式，这样可以方便的做时序或者截面的处理。矩阵话后更容易处理。

    Parameters
    ----------
    session : DolphinDB session
        DolphinDB 会话对象
    instruments : Union[str, List[str]]
        证券代码或证券代码列表
    fields : Union[List[str],str,Dict]
        Qlib 格式的表达式或表达式列表
        如果fields为字典则key-表达式,value-别名
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

    normalized_expr, base_fields, is_pure_fields = normalize_fields_to_ddb(fields)
    reversed_expr: Dict = {v: k for k, v in normalized_expr.items()}

    _freq: str = "daily" if freq == "day" else "min"
    feature_schema: QlibTableSchema = getattr(QlibTableSchema(), f"feature_{_freq}")()

    db_name: str = feature_schema.db_name
    table_name: str = feature_schema.table_name

    # 获取实际查询范围
    date_utils: TradeDateUtils = TradeDateUtils(session, freq)
    start_time, end_time = date_utils.get_locate_date(start_time, end_time)

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

    # 判断是否为纯字段表达式（如$close, $open），如果是则直接查询表
    if is_pure_fields:
        # 基础字段如果缺失则使用0填充
        base_fields: List[str] = build_field_expr(
            session, db_name, table_name, base_fields
        )
        # 创建基础查询语句部分
        base_query = (
            session.loadTable(table_name, f"dfs://{db_name}")
            .select(["code", "date"] + base_fields)
            .where(f"date between pair({start_time},{end_time})")
        )

        # 根据instruments类型选择不同的查询方式
        if isinstance(instruments, list):
            # 如果是列表，直接使用in条件过滤
            data: pd.DataFrame = (
                base_query.where(f"code in instruments").sort(["date", "code"]).toDF()
            )

        elif isinstance(instruments, dict):
            # 如果是字典，使用conditionalFilter进行复杂的日期-股票映射过滤
            # 先在DolphinDB中创建日期-股票映射,用以兼容spans
            session.run(
                "codeRangeFilter = createDateStockMapping(start_time,end_time,instruments)"
            )

            # 使用conditionalFilter应用复杂过滤条件
            data: pd.DataFrame = (
                base_query.where("conditionalFilter(code, date, codeRangeFilter)")
                .sort(["date", "code"])
                .toDF()
            )

    else:

        # 使用mr防止因子计算时OOM
        ddb_expr = f"""        
        FeatureEngineeringByDate(instruments,expressions,baseFields,start_time,end_time,"{db_name}","{table_name}")
        """
        # data: pd.DataFrame = session.run(ddb_expr)
        data: Dict[str, List] = session.run(ddb_expr)
        try:
            data: Dict[str, pd.DataFrame] = {
                k: pd.DataFrame(data=v[0], index=v[1], columns=v[2])
                for k, v in data.items()
            }
        except IndexError as e:
            # 可能是data为空
            raise IndexError(f"传入的data数据为空: {e}")
        data: pd.DataFrame = pd.concat(data)

    if not isinstance(data, pd.DataFrame):
        raise ValueError("查询结果不是 DataFrame 格式")

    # 格式化结果
    if not data.empty:
        try:
            if isinstance(data.index, pd.MultiIndex):
                # 根据 pandas 版本决定 stack 的参数
                if version.parse(pd.__version__) < version.parse("1.5.0"):
                    # 旧版本 pandas 不支持 future_stack，但支持 dropna=False
                    data: pd.DataFrame = (
                        data.unstack(level=0)
                        .stack(level=0, dropna=False)
                        .swaplevel(0, 1)
                    )
                else:
                    # 新版本 pandas 使用 future_stack 或依赖默认行为
                    data: pd.DataFrame = (
                        data.unstack(level=0)
                        .stack(level=0, future_stack=True)
                        .swaplevel(0, 1)
                    )
            else:
                data: pd.DataFrame = data.set_index(["code", "date"])
        except KeyError:
            raise KeyError("查询结果缺少 'code' 或 'date' 列")

        data.index.names = ["instrument", "datetime"]

    data = data.sort_index()
    if is_pure_fields:
        return data
    # 重命名columns将$改为去掉$或根据已有的字典进行重命名
    try:
        aliases_order = list(normalized_expr.values())
        # 只保留实际存在的 alias，避免 reindex 新增空列
        ordered_aliases = [c for c in aliases_order if c in data.columns]
        # 若希望严格要求所有列必须存在，可以改为：
        # missing = [c for c in aliases_order if c not in data.columns]; if missing: raise KeyError(...)
        data = data.reindex(columns=ordered_aliases)
    except Exception as e:
        # 出错则保持原始列顺序
        raise e

    # 最后再重命名 alias -> 输出列名
    data = data.rename(columns=reversed_expr)

    return data


# # 用于兼容qlib的表达式
# def ddb_compute_features(
#     session,
#     instruments: Union[List, str],
#     fields: Union[List, str],
#     start_time: str,
#     end_time: str,
#     db_name: str,
#     table_name: str,
#     mr_by_code: bool = False,
#     column_name: str = "date",
#     freq: str = "day",
# ) -> pd.DataFrame:
#     """
#     计算DolphinDB中的特征。
#     --------
#     因子处理流程与原始qlib差不多类似df.groupby("code").apply(lambda x:expression(x))的模式;
#     缺点：这种模式在处理横截面表达式时就不行

#     此函数从DolphinDB数据库中检索和计算特定时间范围内的特定特征。

#     :param session: DolphinDB会话对象
#     :type session: DDB会话对象

#     :param instruments: 代码列表或单个代码字符串（例如股票代码）
#     :type instruments: Union[List, str]

#     :param fields: 要获取的字段列表或单个字段字符串(Qlib原生表达式)
#     :type fields: Union[List, str]

#     :param start_time: 数据开始时间
#     :type start_time: str

#     :param end_time: 数据结束时间
#     :type end_time: str

#     :param db_name: DolphinDB数据库名称
#     :type db_name: str

#     :param table_name: DolphinDB表名
#     :type table_name: str

#     :param mr_by_code: 是否按代码进行MapReduce操作，默认为False。注意：社区版DolphinDB在mr_by_code=True时可能出现OOM问题
#     :type mr_by_code: bool, optional

#     :param column_name: 时间列的名称，默认为"date"
#     :type column_name: str, optional

#     :param freq: 数据频率，可选"day"或其他值（如"min"），默认为"day"
#     :type freq: str, optional

#     :returns: 包含计算特征的DataFrame，按代码和日期排序
#     :rtype: pd.DataFrame
#     """
#     if isinstance(fields, str):
#         fields = [fields]

#     if isinstance(instruments, str):
#         instruments = [instruments]

#     fmt: str = "%Y.%m.%d" if freq == "day" else "%Y.%m.%d %H:%M:%S"
#     start_time: pd.Timestamp = pd.to_datetime(start_time).strftime(fmt)
#     end_time: pd.Timestamp = pd.to_datetime(end_time).strftime(fmt)

#     # 标准化qlib表达
#     exprs: List[str] = normalize_cache_fields([str(column) for column in fields])
#     # escape_backslash=True, 后续用于parseExpr解析
#     # 将qlib表达式转换为DolphinDB表达式
#     normalized_expr: List[str] = [
#         adapt_qlib_expr_syntax_for_ddb(column, OPERATOR_MAPPING, True)
#         for column in exprs
#     ]

#     # 使用mr_by_code会快很多,但是社区版ddb会出现OOM出现,所有尽量使用mr_by_code=false
#     return session.run(
#         f"""FeatureEngineering({instruments},dict({normalized_expr},{exprs}),{start_time},{end_time},"{db_name}","{table_name}",{int(mr_by_code)}$bool,"{column_name}")"""
#     ).sort_values(["code", "date"])


##############################################################################################################################


class TradeDateUtils:

    def __init__(self, session: ddb.Session, freq: str):
        db_path: str = f"dfs://{QlibTableSchema.calendar().db_name}"
        tb_path: str = QlibTableSchema.calendar().table_name
        self._calendar: np.ndarray = (
            session.loadTable(tb_path, db_path)
            .exec("TRADE_DAYS")
            .sort("TRADE_DAYS")
            .toDF()
        )
        self._calendar_index: Dict = dict(
            zip(self._calendar, np.arange(len(self._calendar)))
        )

    def get_locate_date_by_idx(
        self, start_idx: int, end_idx: int
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """根据索引获取交易日历"""
        if (not isinstance(start_idx, int)) or (not isinstance(end_idx, int)):
            raise TypeError("索引必须为整数")

        if start_idx > end_idx:
            raise ValueError("开始索引不能大于结束索引")

        if start_idx < 0 or end_idx < 0:
            raise ValueError("索引不能为负数")

        if start_idx > len(self._calendar) or end_idx > len(self._calendar):
            raise ValueError("索引超出范围")

        start_date: pd.Timestamp = self._calendar[start_idx]
        end_date: pd.Timestamp = self._calendar[end_idx]
        return start_date, end_date

    def get_locate_date(
        self, start_dt: Union[int, str], end_dt: Union[int, str]
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """根据日期获取交易日历"""

        # 处理整数索引情况
        if isinstance(start_dt, int) or isinstance(end_dt, int):
            return self.get_locate_date_by_idx(
                0 if start_dt is None else start_dt,
                len(self._calendar) - 1 if end_dt is None else end_dt,
            )

        # 处理None值
        cal_min = pd.Timestamp(self._calendar.min())
        cal_max = pd.Timestamp(self._calendar.max())

        if start_dt is None:
            start_dt = cal_min
        else:
            start_dt = pd.Timestamp(start_dt)

        if end_dt is None:
            end_dt = cal_max
        else:
            end_dt = pd.Timestamp(end_dt)

        # 查找交易日期
        if start_dt not in self._calendar_index:
            try:
                idx = bisect.bisect_left(self._calendar, start_dt)
                if idx >= len(self._calendar):
                    raise IndexError(f"开始日期 {start_dt} 超出交易日历范围")
                start_dt = self._calendar[idx]
            except IndexError as index_e:
                raise IndexError(f"开始日期 {start_dt} 超出交易日历范围") from index_e

        if end_dt not in self._calendar_index:
            try:
                idx = bisect.bisect_right(self._calendar, end_dt) - 1
                if idx < 0:
                    raise IndexError(f"结束日期 {end_dt} 超出交易日历范围")
                end_dt = self._calendar[idx]
            except IndexError as index_e:
                raise IndexError(f"结束日期 {end_dt} 超出交易日历范围") from index_e

        # 确保开始日期不晚于结束日期
        if start_dt > end_dt:
            raise ValueError(f"开始日期 {start_dt} 不能晚于结束日期 {end_dt}")

        return start_dt, end_dt


def construct_ddb_sql_eval(
    db_name: str,
    table_name: str,
    select,
    where=None,
    groupBy=None,
    csort=None,
    having=None,
    orderBy=None,
    groupFlag: int = 1,
    ascSort=1,
    ascOrder=1,
    limit=None,
    exec: bool = False,
) -> str:
    """
    生成 DolphinDB 查询脚本:
        tb = loadTable("dfs://{db_name}","{table_name}");
        query_expr = sql(
            select=...,
            from=tb,
            where=...,
            ...仅输出提供的参数...
            groupFlag=...,
            ascSort=...,
            ascOrder=...,
            exec=...
        );
        query_expr.eval()

    仅对传入的非 None 参数输出对应行；未提供的不出现。
    csort 仅在 groupFlag==0 时并且传入不为 None 才输出。
    select 可为字符串或序列(序列自动包装为 ANY 向量 <[a,b,c]> )
    其他可为字符串 / 序列(转 ANY) / None
    ascSort, ascOrder 可为 int 或序列(转 [..])
    limit: int -> N, (s,e)-> s:e
    """
    if not db_name or not table_name:
        raise ValueError("必须提供 db_name 与 table_name")

    def _to_any_vector(val):
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            if not val:
                return None
            return "<[" + ",".join(str(x) for x in val) + "]>"
        return str(val)

    def _to_order_vec(v):
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            if not v:
                return None
            return "[" + ",".join(str(int(x)) for x in v) + "]"
        return str(int(v))

    def _to_limit(v):
        if v is None:
            return None
        if isinstance(v, int):
            return str(v)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            s, e = v
            return f"{int(s)}:{int(e)}"
        raise ValueError("limit 只能为 None / int / (start,end)")

    select_code = _to_any_vector(select)
    where_code = _to_any_vector(where)
    group_code = _to_any_vector(groupBy)
    csort_code = (
        _to_any_vector(csort) if (groupFlag == 0 and csort is not None) else None
    )
    having_code = _to_any_vector(having)
    order_code = _to_any_vector(orderBy)
    asc_sort_code = _to_order_vec(ascSort)
    asc_order_code = _to_order_vec(ascOrder)
    limit_code = _to_limit(limit)
    exec_code = "true" if exec else "false"

    def add_line(name, value, wrap_sqlCol=False):
        if value is None:
            return None
        if wrap_sqlCol and not str(value).startswith("sqlCol("):
            return f"{name}=sqlCol({value}),"
        return f"{name}={value},"

    lines = []
    lines.append(add_line("select", select_code))
    lines.append("from=tb,")
    if where_code is not None:
        lines.append(add_line("where", where_code))
    if group_code is not None:
        lines.append(add_line("groupBy", group_code))
    if csort_code is not None:
        lines.append(add_line("csort", csort_code))
    if having_code is not None:
        lines.append(add_line("having", having_code))
    if order_code is not None:
        lines.append(add_line("orderBy", order_code))
    lines.append(f"groupFlag={int(groupFlag)},")
    lines.append(f"ascSort={asc_sort_code or '1'},")
    lines.append(f"ascOrder={asc_order_code or '1'},")
    if limit_code is not None:
        lines.append(f"limit={limit_code},")
    lines.append(f"exec={exec_code}")

    lines = [l for l in lines if l is not None]

    sql_block = "query_expr = sql(\n    " + "\n    ".join(lines) + "\n);"

    script = f"""
    tb = loadTable("dfs://{db_name}","{table_name}");
    {sql_block}
    query_expr.eval()
    """.strip()
    return script


# def get_query_date_range(
#     session: ddb.Session, start_dt, end_dt
# ) -> Tuple[pd.Timestamp, pd.Timestamp]:
#     """获取查询的日期范围

#     根据提供的开始和结束日期，从日历表中确定实际的查询日期范围。
#     如果未指定开始或结束日期，则使用日历表中的最早或最晚交易日。

#     :param session: DolphinDB会话对象
#     :type session: ddb.Session
#     :param start_dt: 开始日期，可以是以下类型：
#                     - None：使用日历表中的最早日期
#                     - int：作为日历表中的索引位置
#                     - str或pd.Timestamp：具体的日期值
#     :type start_dt: Union[str, int, pd.Timestamp, None]
#     :param end_dt: 结束日期，可以是以下类型：
#                   - None：使用日历表中的最晚日期
#                   - int：作为日历表中的索引位置
#                   - str或pd.Timestamp：具体的日期值
#     :type end_dt: Union[str, int, pd.Timestamp, None]

#     :return: 处理后的开始日期和结束日期，转换为pandas Timestamp格式
#     :rtype: Tuple[pd.Timestamp, pd.Timestamp]
#     """

#     db_path: str = f"dfs://{QlibTableSchema.calendar().db_name}"
#     tb_path: str = QlibTableSchema.calendar().table_name

#     tb: pd.DataFrame = (
#         session.loadTable(tb_path, db_path)
#         .select("TRADE_DAYS")
#         .sort("TRADE_DAYS")
#         .toDF()
#     )
#     dates: pd.DatetimeIndex = pd.to_datetime(tb["TRADE_DAYS"].values).sort_values()

#     if start_dt == None:
#         start_dt: pd.Timestamp = dates.min()

#     if end_dt == None:
#         end_dt: pd.Timestamp = dates.max()

#     if isinstance(start_dt, int) or isinstance(end_dt, int):

#         size: int = len(dates)
#         if end_dt > size:
#             raise ValueError(f"结束日期超出交易日历范围: {end_dt} > {size}")

#         return dates[start_dt], dates[end_dt]

#     else:

#         start_dt: pd.Timestamp = dates.asof(start_dt)
#         end_dt: pd.Timestamp = dates.asof(end_dt)

#         # 检查开始日期
#         if start_dt is np.nan:
#             raise ValueError(f"开始日期 {start_dt} 不是有效的交易日")

#         # 检查结束日期
#         if start_dt is np.nan:
#             raise ValueError(f"结束日期 {end_dt} 不是有效的交易日")

#         return start_dt, end_dt


###################################################################################################################################
#                                       表达式转换函数
###################################################################################################################################
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
