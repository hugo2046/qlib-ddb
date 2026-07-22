"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-28 14:00:40
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-28 14:08:23
Description:
"""

import bisect
import functools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import dolphindb as ddb
import numpy as np
import pandas as pd
from packaging import version

from ....log import get_module_logger
from ....utils import normalize_cache_fields,remove_fields_space
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


def _sort_ddb_scripts(scripts: Iterable[Path]) -> List[Path]:
    """对 ddb_scripts 下脚本排序，返回与文件系统无关的确定性加载顺序。

    ⚠️ 顺序约束（已通过实际脚本依赖核查验证）：
    - ``ops.dos`` 定义了 ``Slope``/``Rsquare``/``Resi`` 等基础算子；
    - ``qlib158Alpha.dos`` 跨文件引用 ``Slope``（首次使用见 qlib158Alpha.dos:154）；
    - 故 ``ops.dos`` 必须先于其它脚本加载。

    ``Path.glob()`` 返回顺序依赖文件系统 ``readdir``（macOS APFS 与 Linux ext4
    顺序不同）。若直接遍历 glob 结果，Linux 上 ``qlib158Alpha.dos`` 可能先于
    ``ops.dos`` 执行，触发
    ``Syntax Error: [line #154] Cannot recognize the token Slope``，
    导致 qlib 初始化失败。

    参数:
        scripts: 待排序的脚本路径集合（通常来自 ``script_path.glob("*.dos")``）。

    返回:
        排序后的路径列表：``ops.dos`` 置首，其余按文件名字母序。
    """
    # ops.dos 置首（键 0），其余按文件名字母序（键 1）
    return sorted(scripts, key=lambda p: (0 if p.name == "ops.dos" else 1, p.name))


# 核心脚本：查询引擎必需，init 即加载（ops.dos 必须置首，见 _sort_ddb_scripts）
CORE_DDB_SCRIPTS: Tuple[str, ...] = (
    "ops.dos",
    "featureEngineering.dos",
    "prepareInstruments.dos",
)

# alpha 因子库（合计约 119KB 脚本）：仅当字段引用对应前缀函数时才按需加载，
# 避免每次 qlib.init 都让服务器解析大量通常用不到的脚本（社区版 2 核/8GB）
ALPHA_LIB_SCRIPTS: Dict[str, str] = {
    "gtjaalpha": "gtja191Alpha.dos",
    "qlib158alpha": "qlib158Alpha.dos",
    "wqalpha": "wq101alpha.dos",
}

# 会话对象上记录已加载 alpha 库前缀集合的属性名（函数注册是会话级状态）
_LOADED_LIBS_ATTR = "_qlib_loaded_alpha_libs"


def register_ddb_functions_to_qlib(
    session: ddb.Session, preload_alpha_libs: Union[bool, None] = None
) -> None:
    """
    在 DolphinDB 会话中注册与 qlib 对应的自定义函数

    这些函数实现了 qlib 中的特定操作，使 DolphinDB 能够兼容 qlib 的表达式计算。
    默认仅加载核心脚本（ops/featureEngineering/prepareInstruments）；三个
    alpha 因子库由 :func:`ensure_alpha_libs_loaded` 在字段引用时按需加载。

    参数:
    - session: DolphinDB 会话实例
    - preload_alpha_libs: True 时恢复历史行为（init 全量加载 alpha 库）；
      None 时读取配置 ``C["ddb_preload_alpha_libs"]``（默认 False）
    """
    if preload_alpha_libs is None:
        try:
            from ....config import C

            preload_alpha_libs = bool(C.get("ddb_preload_alpha_libs", False))
        except Exception:
            preload_alpha_libs = False

    # 在会话中执行函数定义脚本
    # ⚠️ 必须用 _sort_ddb_scripts 确定加载顺序，不能直接遍历 glob（其顺序依赖
    #    文件系统 readdir，跨平台不一致；详见 _sort_ddb_scripts 的文档）。
    script_path = Path(__file__).parent / "ddb_scripts"
    if preload_alpha_libs:
        scripts = _sort_ddb_scripts(script_path.glob("*.dos"))
        loaded_libs = set(ALPHA_LIB_SCRIPTS)
    else:
        scripts = _sort_ddb_scripts(script_path / name for name in CORE_DDB_SCRIPTS)
        loaded_libs = set()
    for script_file in scripts:
        session.runFile(script_file)
    setattr(session, _LOADED_LIBS_ATTR, loaded_libs)
    get_module_logger("ddb_features").info("已注册 qlib 兼容函数到 DolphinDB 会话")


def ensure_alpha_libs_loaded(session: ddb.Session, fields) -> None:
    """按字段引用惰性加载 alpha 因子库（每会话每库仅加载一次）。

    以大小写不敏感的前缀匹配（gtjaAlpha/qlib158Alpha/WQAlpha）扫描原始
    字段文本；命中且未加载时 ``runFile`` 对应脚本并在会话上打标记。

    :param session: DolphinDB 会话实例
    :param fields: 原始字段（str/list/dict 均可，转文本扫描）
    """
    loaded = getattr(session, _LOADED_LIBS_ATTR, None)
    if loaded is None:
        loaded = set()
        setattr(session, _LOADED_LIBS_ATTR, loaded)
    if loaded >= set(ALPHA_LIB_SCRIPTS):
        return
    text = str(fields).lower()
    script_path = Path(__file__).parent / "ddb_scripts"
    for prefix, script_name in ALPHA_LIB_SCRIPTS.items():
        if prefix in loaded or prefix not in text:
            continue
        session.runFile(script_path / script_name)
        loaded.add(prefix)
        get_module_logger("ddb_features").info(f"已按需加载 alpha 因子库 {script_name}")


def _load_all_alpha_libs(session: ddb.Session) -> None:
    """加载全部尚未加载的 alpha 因子库（未识别函数兜底重试用）。"""
    loaded = getattr(session, _LOADED_LIBS_ATTR, None)
    if loaded is None:
        loaded = set()
        setattr(session, _LOADED_LIBS_ATTR, loaded)
    script_path = Path(__file__).parent / "ddb_scripts"
    for prefix, script_name in ALPHA_LIB_SCRIPTS.items():
        if prefix not in loaded:
            session.runFile(script_path / script_name)
            loaded.add(prefix)


##################################################################################################################

# 回看窗口启发式解析：独立整数字面量（排除标识符与小数的组成部分，
# 如 gtjaAlpha191_001 中的 191 不会被误判为窗口）
_STANDALONE_INT_PATTERN = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
# 负数窗口（未来引用，如 Ref($close,-2)）仅以函数参数形式出现，
# 避免把减法运算（如 "close-1"）误判为未来引用
_FUTURE_INT_PATTERN = re.compile(r",\s*-(\d+)")
# 兜底扫描到的整数超过此上界视为普通数值常量（如缩放因子 $volume/1000000），
# 而非滚动窗口，避免误判成百万日回看导致 shiftTradeDays 外扩到不合理区间。
# 约 8 年交易日，远超常见滚动窗口（年化 252、双年 504 等）
_MAX_FALLBACK_WINDOW: int = 2000


@functools.lru_cache(maxsize=4096)
def get_expression_extended_window(expr: str, default_lookback: int) -> Tuple[int, int]:
    """解析 qlib 表达式所需的（向前, 向后）外扩交易日数。

    与文件后端 ``LocalExpressionProvider.file_expression`` 的取前序期语义
    对齐：优先复用 qlib 算子树的 ``get_extended_window_size``（递归处理
    嵌套算子、双臂算子取 max、以及 ``Ref($close,-2)`` 这类未来引用的向后
    外扩）。qlib 无法实例化的表达式（DDB 专属 alpha 库函数等）退回启发式：
    正则扫描独立整数字面量当窗口（超过 :data:`_MAX_FALLBACK_WINDOW` 的整数
    视为数值常量而非窗口予以剔除），扫不到时用 ``default_lookback`` 兜底；
    真正非法的表达式仍会在 DDB 端 parseExpr 阶段报错（原有行为）。

    :param expr: 原始 qlib 表达式（含 ``$`` 字段引用）
    :param default_lookback: 启发式扫不到窗口时的兜底回看交易日数
    :return: ``(向前外扩交易日数, 向后外扩交易日数)``
    """
    try:
        from ....config import C
        from ....utils import parse_field
        from ...base import Feature, PFeature
        from ...ops import Operators, register_all_ops

        # 未经 qlib.init 的独立调用场景（如离线测试）惰性注册默认算子
        if not getattr(Operators, "_ops", None):
            register_all_ops(C)
        # 安全说明：与 qlib 核心 ExpressionProvider.get_expression_instance
        # 完全同款的 eval 解析（同一信任模型：字段表达式来自使用者配置），
        # 且此处限定了命名空间仅暴露 Operators/Feature/PFeature
        instance = eval(
            parse_field(expr),
            {"Operators": Operators, "Feature": Feature, "PFeature": PFeature},
        )
        lft_etd, rght_etd = instance.get_extended_window_size()
        return int(lft_etd), int(rght_etd)
    except Exception:
        # 剔除超过上界的整数（视为数值常量而非窗口）；剩余最大者为回看窗口
        ints = [
            n
            for n in (int(s) for s in _STANDALONE_INT_PATTERN.findall(expr))
            if n <= _MAX_FALLBACK_WINDOW
        ]
        lft_etd = max(ints) if ints else int(default_lookback)
        futures = [
            n
            for n in (int(s) for s in _FUTURE_INT_PATTERN.findall(expr))
            if n <= _MAX_FALLBACK_WINDOW
        ]
        return lft_etd, max(futures) if futures else 0


def batch_extended_window(
    exprs: Iterable[str], default_lookback: int
) -> Tuple[int, int]:
    """对一批表达式取（向前, 向后）外扩交易日数的最大值。

    同批表达式共享同一份基础数据面板，故按批取 max 外扩。

    :param exprs: 原始 qlib 表达式集合
    :param default_lookback: 单表达式解析失败时的兜底回看交易日数
    :return: ``(向前外扩交易日数, 向后外扩交易日数)``
    """
    lft_etd, rght_etd = 0, 0
    for expr in exprs:
        lft, rght = get_expression_extended_window(expr, default_lookback)
        lft_etd, rght_etd = max(lft_etd, lft), max(rght_etd, rght)
    return lft_etd, rght_etd


def normalize_fields_to_ddb(
    fields: Union[str, List[str], Dict],
) -> Tuple[Dict, List, bool,Dict]:
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

    # 这个字典将别名映射回最原始的表达式
    alias_to_origin_expr_map = {alias: remove_fields_space(expr) for expr, alias in fields_od.items()}
    
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

    return normalized_expr, base_fields, is_pure_fields, alias_to_origin_expr_map


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
    from .utils import get_table_columns

    # 经进程内缓存读取列名（表结构仅随 DDL 变更），预热后 0 RPC
    table_columns: List[str] = get_table_columns(session, f"dfs://{db_name}", table_name)

    return [col if col in table_columns else f"0 as {col}" for col in base_fields]


def apply_spans_mask(data: pd.DataFrame, spans: Dict[str, List]) -> pd.DataFrame:
    """按成分股 spans（入池/出池区间）对结果面板做行掩码。

    用于 ``fetch_features_from_ddb`` 非纯字段分支的 Python 侧兜底过滤：
    计算分支向 ``FeatureEngineeringByDate`` 传键列表全量计算（历史上因
    mr worker 端还原不了 ``conditionalFilter`` 所需的 dict，协议保持不变），
    spans 过滤在此处补齐。先算后掩码的语义与原版 qlib
    ``inst_calculator``（data.py 中按 spans 做 mask）一致。

    :param data: 以 ``(instrument, datetime)`` 为 MultiIndex 的结果面板。
    :type data: pd.DataFrame
    :param spans: 股票代码到入池/出池区间列表的映射，
        形如 ``{code: [(入池日, 出池日), ...]}``；不在映射中的代码整体剔除。
    :type spans: Dict[str, List]
    :return: 仅保留各代码在其 spans 区间内行的面板。
    :rtype: pd.DataFrame
    """
    if data.empty:
        return data

    # 展开 spans 为 (instrument, begin, end) 边界表，与行索引 merge 后向量化判断
    bounds = pd.DataFrame(
        [(code, begin, end) for code, span_list in spans.items() for begin, end in span_list],
        columns=["instrument", "begin", "end"],
    )
    idx = pd.DataFrame(
        {
            "row": np.arange(len(data)),
            "instrument": data.index.get_level_values("instrument"),
            "datetime": data.index.get_level_values("datetime"),
        }
    )
    merged = idx.merge(bounds, on="instrument", how="inner")
    hit_rows = merged.loc[
        (merged["datetime"] >= merged["begin"]) & (merged["datetime"] <= merged["end"]),
        "row",
    ].unique()

    mask = np.zeros(len(data), dtype=bool)
    mask[hit_rows] = True
    return data[mask]


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

    normalized_expr, base_fields, is_pure_fields, alias_to_origin_expr_map = normalize_fields_to_ddb(fields)

    _freq: str = "daily" if freq == "day" else "min"
    feature_schema: QlibTableSchema = getattr(QlibTableSchema(), f"feature_{_freq}")()

    db_name: str = feature_schema.db_name
    table_name: str = feature_schema.table_name

    # 交易日对齐并格式化为 DDB 日期字面量
    start_time, end_time = _resolve_query_window(session, freq, start_time, end_time)

    # 转换为列表格式
    if isinstance(instruments, str):
        instruments = [instruments]

    # 防御性检查：filter_pipe 可能把 instruments 全部过滤掉，
    # 空 instruments 直接早退，避免进入 DDB 端空数据兜底路径
    if not instruments:
        return pd.DataFrame()

    # 字段若引用 alpha 因子库函数，按需加载对应 .dos（每会话仅一次）
    ensure_alpha_libs_loaded(session, fields)

    if is_pure_fields:
        data = _fetch_pure_fields(
            session, instruments, base_fields, db_name, table_name, start_time, end_time
        )
    else:
        # 滚动/引用算子按 qlib 算子树外扩查询窗口（DDB 端计算后截断回请求
        # 区间），对齐文件后端 get_extended_window_size 的取前序期语义，
        # 避免 Mean($close,20) 在 start_time 后前 19 个交易日全为 NaN
        try:
            from ....config import C

            default_lookback = int(C.get("ddb_lookback_default", 252))
        except Exception:
            default_lookback = 252
        lookback_days, right_days = batch_extended_window(
            alias_to_origin_expr_map.values(), default_lookback
        )
        raw = _compute_expressions(
            session, instruments, normalized_expr, base_fields, db_name, table_name,
            start_time, end_time, lookback_days, right_days,
        )
        # DDB 端在 baseData 为空时会返回空 dict（兜底路径），需要在此处早返回
        # 避免 pd.concat({}) 报 "No objects to concatenate"
        if raw is None:
            return pd.DataFrame()
        data = _computed_dict_to_panel(raw)

    return _format_result_panel(
        data, is_pure_fields, instruments, normalized_expr, alias_to_origin_expr_map
    )


def _resolve_query_window(
    session: ddb.Session,
    freq: str,
    start_time: Union[pd.Timestamp, int],
    end_time: Union[pd.Timestamp, int],
) -> Tuple[str, str]:
    """按交易日历对齐查询区间，并格式化为 DDB 日期字面量。

    ⚠️ 性能约定：日期以字面量内联进查询脚本，不再单独
    ``session.run`` 上传 ``start_time``/``end_time`` 服务器变量
    （曾经每批字段多付一次网络往返）。

    :param session: DolphinDB 会话
    :param freq: 数据频率（"day" / "min"）
    :param start_time: 开始时间（时间戳或日历下标）
    :param end_time: 结束时间（时间戳或日历下标）
    :return: (开始, 结束) 的 DDB 日期字面量字符串
    """
    date_utils: TradeDateUtils = TradeDateUtils(session, freq)
    start_time, end_time = date_utils.get_locate_date(start_time, end_time)

    fmt: str = "%Y.%m.%d" if freq == "day" else "%Y.%m.%d %H:%M:%S"
    return (
        pd.to_datetime(start_time).strftime(fmt),
        pd.to_datetime(end_time).strftime(fmt),
    )


def _fetch_pure_fields(
    session: ddb.Session,
    instruments: Union[List[str], Dict],
    base_fields: List[str],
    db_name: str,
    table_name: str,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """纯字段分支：单条 SQL 直查特征表（含 spans dict 的 conditionalFilter 过滤）。

    复杂值（instruments）走变量上传、标量日期走字面量内联——参见
    dolphindb_skill 对「变量上传优于字符串拼接」的建议（doc_7605/doc_5576）。
    dict（成分股 spans）分支把日期-股票映射与主查询合并为一次 ``session.run``，
    消除历史上独立的 ``createDateStockMapping`` 往返。

    :return: 长表 DataFrame（columns: code, date, 各基础字段）
    """
    session.upload({"instruments": instruments})

    # 基础字段如果缺失则使用0填充
    select_fields: List[str] = build_field_expr(session, db_name, table_name, base_fields)
    select_clause: str = ", ".join(["code", "date"] + select_fields)

    if isinstance(instruments, dict):
        # spans dict：先在 DDB 中创建日期-股票映射，再用 conditionalFilter 过滤（同一脚本内完成）
        script = f"""
        codeRangeFilter = createDateStockMapping({start_time},{end_time},instruments);
        select {select_clause} from loadTable("dfs://{db_name}","{table_name}")
        where date between pair({start_time},{end_time})
            and conditionalFilter(code, date, codeRangeFilter)
        order by date, code
        """
    else:
        script = f"""
        select {select_clause} from loadTable("dfs://{db_name}","{table_name}")
        where date between pair({start_time},{end_time}) and code in instruments
        order by date, code
        """
    return session.run(script)


def _compute_expressions(
    session: ddb.Session,
    instruments: Union[List[str], Dict],
    normalized_expr: Dict[str, str],
    base_fields: List[str],
    db_name: str,
    table_name: str,
    start_time: str,
    end_time: str,
    lookback_days: int = 0,
    right_days: int = 0,
) -> Union[pd.DataFrame, None]:
    """计算表达式分支：经 ``FeatureEngineeringByDate`` 服务器端计算。

    DDB 端按内存估算自动选择整段单次计算或按日期分段顺序计算（防 OOM）；
    每次查询按 ``lookback_days``/``right_days`` 外扩窗口，计算后截断回
    请求区间，保证滚动算子在区间头部、未来引用在区间尾部不缺窗口。

    ⚠️ dict（成分股 spans）时不把 dict 传给 FeatureEngineeringByDate：
    历史上其内部经 mr 分布式执行，worker 端还原不了 conditionalFilter
    所需的 dict。现保持同一协议——传键列表全量计算，spans 过滤在拿到
    结果后由 apply_spans_mask 在 Python 侧补齐。

    :param lookback_days: 向前外扩的交易日数（滚动算子回看窗口）
    :param right_days: 向后外扩的交易日数（未来引用，如标签 Ref($close,-2)）
    :return: 服务器返回的原始 ``{alias: [值矩阵(dates×codes), 日期, 代码]}``；
        DDB 返回空 dict（baseData 为空的兜底路径）时返回 None
    """
    session.upload(
        {
            "instruments": instruments,
            "expressions": normalized_expr,
            "baseFields": base_fields,
        }
    )

    inst_expr = "keys(instruments)" if isinstance(instruments, dict) else "instruments"
    try:
        from ....config import C

        days_step = int(C.get("ddb_days_step", 252))
    except Exception:
        days_step = 252
    ddb_expr = f"""
    FeatureEngineeringByDate({inst_expr},expressions,baseFields,{start_time},{end_time},"{db_name}","{table_name}",{days_step},{lookback_days},{right_days})
    """
    def _log_failure() -> None:
        # 记录排查上下文；instruments 可能上千个，只记数量与头部样本
        inst_list = list(instruments)
        get_module_logger("ddb_features").error(
            f"DolphinDB 因子计算失败: db={db_name}/{table_name}, "
            f"时间范围={start_time}~{end_time}, "
            f"instruments 共{len(inst_list)}个(头部样本: {inst_list[:5]}), "
            f"expressions={normalized_expr}, baseFields={base_fields}"
        )

    try:
        data: Dict[str, List] = session.run(ddb_expr)
    except Exception as e:
        # 惰性加载兜底：未识别函数可能是绕过字段扫描调用的 alpha 函数，
        # 全量加载 alpha 库后重试一次
        if "Cannot recognize the token" not in str(e):
            _log_failure()
            raise RuntimeError(f"DolphinDB 因子计算失败: {e}") from e
        _load_all_alpha_libs(session)
        try:
            data: Dict[str, List] = session.run(ddb_expr)
        except Exception as retry_e:
            _log_failure()
            raise RuntimeError(f"DolphinDB 因子计算失败: {retry_e}") from retry_e

    if not data:
        return None
    return data


def _legacy_reshape(data: Dict[str, List]) -> pd.DataFrame:
    """旧版重塑路径：concat → unstack → stack → swaplevel（多次全景拷贝）。

    仅作为 :func:`_computed_dict_to_panel` 形状不一致时的运行时兜底，
    并为直构路径的等价性测试提供对照实现。
    """
    try:
        frames: Dict[str, pd.DataFrame] = {
            k: pd.DataFrame(data=v[0], index=v[1], columns=v[2])
            for k, v in data.items()
        }
    except IndexError as e:
        # 可能是data为空
        raise IndexError(f"传入的data数据为空: {e}")
    panel: pd.DataFrame = pd.concat(frames)
    if panel.empty:
        return panel
    # 根据 pandas 版本决定 stack 的参数
    if version.parse(pd.__version__) < version.parse("1.5.0"):
        # 旧版本 pandas 不支持 future_stack，但支持 dropna=False
        panel = panel.unstack(level=0).stack(level=0, dropna=False).swaplevel(0, 1)
    else:
        # 新版本 pandas 使用 future_stack 或依赖默认行为
        panel = panel.unstack(level=0).stack(level=0, future_stack=True).swaplevel(0, 1)
    return panel


def _computed_dict_to_panel(data: Dict[str, List]) -> pd.DataFrame:
    """把 ``{alias: [值矩阵(dates×codes), 日期, 代码]}`` 直构为结果面板。

    ⚠️ 性能关键：旧路径 concat → unstack → stack → swaplevel → sort_index
    要做约 4 次全景拷贝（5000 股 × 多年 × 30 列时数百 MB 的客户端内存churn）；
    此处按 (instrument, datetime) 直接一次分配构造。所有 alias 共享同一
    (dates, codes)（服务器端同一 panel() 产出）；形状不一致（如 DDB 端
    错误占位返回 TABLE）时回退 :func:`_legacy_reshape` 保持旧语义。
    """
    try:
        aliases = list(data.keys())
        first = data[aliases[0]]
        ref_dates = np.asarray(first[1])
        ref_codes = list(first[2])
        expected_shape = (len(ref_dates), len(ref_codes))
        for alias in aliases:
            values, dates, codes = data[alias][0], data[alias][1], data[alias][2]
            if np.asarray(values).shape != expected_shape:
                raise ValueError("值矩阵形状不一致")
            if not np.array_equal(np.asarray(dates), ref_dates) or list(codes) != ref_codes:
                raise ValueError("日期/代码轴不一致")
    except (ValueError, TypeError, IndexError, KeyError, AttributeError):
        return _legacy_reshape(data)

    dates_index = pd.DatetimeIndex(ref_dates)
    codes_arr = np.asarray(ref_codes)
    # 预排序两轴，使输出与旧路径 sort_index() 后的顺序一致
    date_order = np.argsort(dates_index.values, kind="stable")
    code_order = np.argsort(codes_arr, kind="stable")
    index = pd.MultiIndex.from_product(
        [codes_arr[code_order], dates_index[date_order]], names=["instrument", "datetime"]
    )
    columns = {
        alias: np.asarray(data[alias][0])[np.ix_(date_order, code_order)].T.ravel()
        for alias in aliases
    }
    return pd.DataFrame(columns, index=index)


def _format_result_panel(
    data: pd.DataFrame,
    is_pure_fields: bool,
    instruments: Union[List[str], Dict],
    normalized_expr: Dict[str, str],
    alias_to_origin_expr_map: Dict[str, str],
) -> pd.DataFrame:
    """把查询结果整形为 (instrument, datetime) MultiIndex 面板并恢复输出列名。"""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("查询结果不是 DataFrame 格式")

    # 格式化结果：计算分支已由 _computed_dict_to_panel/_legacy_reshape 产出
    # (instrument, datetime) 面板；纯字段分支为长表，需 set_index
    if not data.empty:
        if not isinstance(data.index, pd.MultiIndex):
            try:
                data: pd.DataFrame = data.set_index(["code", "date"])
            except KeyError:
                raise KeyError("查询结果缺少 'code' 或 'date' 列")

        data.index.names = ["instrument", "datetime"]

    data = data.sort_index()
    if is_pure_fields:
        # 纯字段分支的 spans 过滤已由 DDB 端 conditionalFilter 完成，无需掩码
        data.columns = "$" + data.columns
        return data

    # 非纯字段分支：DDB 端按键列表全量计算，此处按 spans 补掩码（见上方注释）
    if isinstance(instruments, dict):
        data = apply_spans_mask(data, instruments)
    # 重命名columns将$改为去掉$或根据已有的字典进行重命名
    aliases_order = list(normalized_expr.values())
    # 只保留实际存在的 alias，避免 reindex 新增空列
    ordered_aliases = [c for c in aliases_order if c in data.columns]
    data = data.reindex(columns=ordered_aliases)
    # 最后再重命名 alias -> 输出列名
    data = data.rename(columns=alias_to_origin_expr_map)

    return data




##############################################################################################################################


class TradeDateUtils:

    # 模块级日历缓存：{(db_path, table_name): (日历数组, 日期->下标索引)}。
    # ⚠️ 性能关键：fetch_features_from_ddb 每批字段都会构造本类，未缓存时
    # 每次都全量下载交易日历（Alpha158 一次 D.features ≈ 6 次下载）。
    # 写路径变更日历表后需调用 ddb_qlib.invalidate_ddb_caches() 失效。
    _calendar_cache: Dict[Tuple[str, str], Tuple[np.ndarray, Dict]] = {}

    def __init__(self, session: ddb.Session, freq: str):
        db_path: str = f"dfs://{QlibTableSchema.calendar().db_name}"
        tb_path: str = QlibTableSchema.calendar().table_name
        cache_key: Tuple[str, str] = (db_path, tb_path)
        cached = self._calendar_cache.get(cache_key)
        if cached is None:
            calendar: np.ndarray = (
                session.loadTable(tb_path, db_path)
                .exec("TRADE_DAYS")
                .sort("TRADE_DAYS")
                .toDF()
            )
            cached = (calendar, dict(zip(calendar, np.arange(len(calendar)))))
            self._calendar_cache[cache_key] = cached
        self._calendar, self._calendar_index = cached

    @classmethod
    def clear_cache(cls) -> None:
        """清空模块级日历缓存（日历表数据变更后调用）。"""
        cls._calendar_cache.clear()

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


###################################################################################################################################
#                                       表达式转换函数
###################################################################################################################################
def adapt_qlib_expr_syntax_for_ddb(
    expr: str, operator_mapping: Dict = OPERATOR_MAPPING, escape_backslash: bool = False
) -> str:
    """
    将 qlib 表达式转换为 DolphinDB 表达式（默认映射时带 lru_cache 记忆化）。

    翻译是纯函数且实现为 ~100 行递归解析，此前每字段每次 fetch 都重跑；
    默认 OPERATOR_MAPPING（identity 判断，覆盖全部真实调用方）时经
    :func:`_adapt_cached` 缓存，自定义映射（dict 不可哈希）则直接解析。

    参数与返回见 :func:`_adapt_qlib_expr_impl`。
    """
    if operator_mapping is OPERATOR_MAPPING:
        return _adapt_cached(expr, escape_backslash)
    return _adapt_qlib_expr_impl(expr, operator_mapping, escape_backslash)


@functools.lru_cache(maxsize=4096)
def _adapt_cached(expr: str, escape_backslash: bool) -> str:
    """默认算子映射下的表达式翻译缓存。"""
    return _adapt_qlib_expr_impl(expr, OPERATOR_MAPPING, escape_backslash)


def _adapt_qlib_expr_impl(
    expr: str, operator_mapping: Dict, escape_backslash: bool = False
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

