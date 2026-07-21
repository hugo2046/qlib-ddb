"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-18 15:05:59
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-25 10:11:05
FilePath:
Description: 用于接入dolphindb
"""

from typing import Iterable, Union, Dict, Mapping, Tuple, List

import pandas as pd
import numpy as np
from qlib.config import C
from qlib.data.cache import H
from qlib.data.data import DBClient
from qlib.data.storage import (
    CalendarStorage,
    CalVT,
    FeatureStorage,
    InstKT,
    InstrumentStorage,
    InstVT,
)
from qlib.log import get_module_logger
from qlib.utils.resam import resam_calendar
from qlib.utils.time import Freq

logger = get_module_logger("db_storage")

TABLE_MAPPING: Dict = {"calendar": "QlibCalendar"}


class DBStorageMixin:

    @property
    def database_uri(self):
        return (
            C["database_uri"]
            if getattr(self, "_database_uri", None) is None
            else self._database_uri
        )

    @property
    def support_freq(self) -> List[str]:
        _v = "_support_freq"
        if hasattr(self, _v):
            return getattr(self, _v)
        freq_l = ["day", "min"]
        freq_l = [Freq(freq) for freq in freq_l]
        setattr(self, _v, freq_l)
        return freq_l

    @property
    def uri(self) -> Tuple[str, str]:
        return self.db_path, self.table_name

    # existsTable 的正向结果缓存：每个存储访问器都会 check()，未缓存时每次
    # 访问都多付一次 RPC。⚠️ 仅缓存 True——表不存在必须持续报错直到其被创建；
    # 写路径经 ddb_qlib.invalidate_ddb_caches() 失效。
    _exists_cache: Dict[Tuple[str, str], bool] = {}

    def exists(self, db_path: str, table_name: str) -> bool:
        key = (db_path, table_name)
        if self._exists_cache.get(key):
            return True
        # ⚠️ 约定：所有 DBClient.session 触点必须持有 session_lock（会话非线程安全）
        with DBClient.session_lock:
            result = bool(DBClient.session.existsTable(db_path, table_name))
        if result:
            self._exists_cache[key] = True
        return result

    def check(self):
        """check self.uri

        Raises
        -------
        ValueError
        """
        if not self.exists(*self.uri):
            raise ValueError(f"{self.storage_name} not exists: {self.uri}")


class DBCalendarStorage(DBStorageMixin, CalendarStorage):

    def __init__(self, freq: str, future: bool, provider_uri: dict = None, **kwargs):
        super(DBCalendarStorage, self).__init__(freq, future, **kwargs)
        self.future = future
        self.enable_read_cache = True  # TODO: make it configurable
        self.region = C["region"]

        # calendars.day
        self.db_path: str = f"dfs://Qlib{self.storage_name.title()}s"
        self.table_name: str = str(self._freq_db)


    # 从这里获取数据
    def _read_calendar(self) -> List[CalVT]:
        # NOTE:
        # if we want to accelerate partial reading calendar
        # we can add parameters like `skip_rows: int = 0, n_rows: int = None` to the interface.
        # Currently, it is not supported for the txt-based calendar

        if not self.exists(self.db_path, self.table_name):
            self._write_calendar(values=[])

        with DBClient.session_lock:
            df: pd.DataFrame = (
                DBClient.session.loadTable(self.table_name, self.db_path).select("*").toDF()
            )
        if df.empty:
            return []

        return df["TRADE_DAYS"].tolist()

    @property
    def _freq_db(self) -> str:
        """the freq to read from file"""
        if not hasattr(self, "_freq_file_cache"):
            freq = Freq(self.freq)
            if freq not in self.support_freq:
                # NOTE: uri
                #   1. If `uri` does not exist
                #       - Get the `min_uri` of the closest `freq` under the same "directory" as the `uri`
                #       - Read data from `min_uri` and resample to `freq`

                freq = Freq.get_recent_freq(freq, self.support_freq)
                if freq is None:
                    raise ValueError(
                        f"can't find a freq from {self.support_freq} that can resample to {self.freq}!"
                    )
            self._freq_file_cache = freq
        return self._freq_file_cache

    def _cached_calendar(self) -> List[CalVT]:
        """经 ``H["c"]`` 缓存读取原始日历（与 ``data`` 共用同一缓存键）。

        ⚠️ 性能关键：``index()``/``__getitem__`` 曾绕过缓存直接
        ``_read_calendar()``，导致数据集对齐期间反复全量下载日历表。
        """
        if self.enable_read_cache:
            key = "orig_file" + str(self.uri)
            if key not in H["c"]:
                H["c"][key] = self._read_calendar()
            return H["c"][key]
        return self._read_calendar()

    @property
    def data(self) -> List[CalVT]:
        self.check()
        _calendar = self._cached_calendar()
        if Freq(self._freq_db) != Freq(self.freq):
            _calendar = resam_calendar(
                np.array(list(map(pd.Timestamp, _calendar))),
                self._freq_db,
                self.freq,
                self.region,
            )
        return _calendar

    def index(self, value: CalVT) -> int:
        self.check()
        calendar = self._cached_calendar()
        return int(np.argwhere(calendar == value)[0])

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, List[CalVT]]:
        self.check()
        return self._cached_calendar()[i]

    def __len__(self) -> int:
        return len(self.data)


class DBInstrumentStorage(DBStorageMixin, InstrumentStorage):
    INSTRUMENT_SEP = "\t"
    INSTRUMENT_START_FIELD = "start_datetime"
    INSTRUMENT_END_FIELD = "end_datetime"
    SYMBOL_FIELD_NAME = "instrument"

    def __init__(self, market: str, freq: str, provider_uri: dict = None, **kwargs):
        super(DBInstrumentStorage, self).__init__(market, freq, **kwargs)

        # 实际调用时D.instruments("pool")
        # 传入的table_name就是instruments.pool
        # self.storage_name = "instruments"此时
        self.db_path: str = f"dfs://Qlib{self.storage_name.title()}s"
        self.table_name: str = market.lower()

    def _read_instrument(self) -> Dict[InstKT, InstVT]:

        self.check()

        # 经 H["i"] 缓存读取（此前每个访问器调用都全量下载 + 行循环重建 dict）；
        # 写路径经 ddb_qlib.invalidate_ddb_caches() 失效
        cache_key = "db_instrument_" + str(self.uri)
        if cache_key in H["i"]:
            # 浅拷贝防止调用方修改缓存本体（值 spans 列表按约定只读）
            return dict(H["i"][cache_key])

        _instruments = dict()

        # sql = f"""SELECT * FROM {self.ddb_table}"""
        # df = DolphinDB.run(sql)
        with DBClient.session_lock:
            df: pd.DataFrame = (
                DBClient.session.loadTable(self.table_name, self.db_path).select("*").toDF()
            )
        for row in df.itertuples(index=False):
            _instruments.setdefault(row[0], []).append((row[1], row[2]))

        H["i"][cache_key] = _instruments
        return dict(_instruments)


    @property
    def data(self) -> Dict[InstKT, InstVT]:
        self.check()
        return self._read_instrument()


    def __getitem__(self, k: InstKT) -> InstVT:
        self.check()
        return self._read_instrument()[k]


    def __len__(self) -> int:
        return len(self.data)


class DBFeatureStorage(DBStorageMixin, FeatureStorage):
    """
    qlib的总体逻辑是获取单只股票数据,然后做算子处理,最后合并.这样使用ddb可能性能上会有问题.OLAP适合直接获取整个数据.
    """

    def __init__(
        self,
        instrument: str,
        field: str,
        freq: str,
        provider_uri: dict = None,
        **kwargs,
    ):
        super(DBFeatureStorage, self).__init__(instrument, field, freq, **kwargs)
        self.field = field
        self.freq = freq.lower()
        # ⚠️ dict 入参为成分股 spans（{code: [(入池, 出池), ...]}），必须保持
        # dict 原型透传给 fetch_features_from_ddb 的 conditionalFilter 分支；
        # 若迭代取键退化为 list，入池/出池日期过滤会静默失效（出池不截断）。
        if isinstance(instrument, str):
            self.instrument = instrument.upper()
        elif isinstance(instrument, dict):
            self.instrument = {k.upper(): v for k, v in instrument.items()}
        else:
            self.instrument = [x.upper() for x in instrument]
        self.db_path = f"Qlib{self.storage_name.title()}s{self.freq.title()}"
        self.table_name = "Features"
        

    @property
    def data(self) -> str:
        return self[:]

    def __getitem__(self, i: Union[int, slice]) -> Union[Tuple[int, float], pd.Series,pd.DataFrame]:
      
        from ..backend.ddb_qlib import fetch_features_from_ddb

        if isinstance(i, (pd.Timestamp,int,str)):
            # 如果单点返回为：(idx,field)

            # 单点查询
            start_time = end_time = i

        elif isinstance(i, slice):
            # 返回为pd.Series,index-idx,values-field
            
            # 区间查询
            start_time = i.start
            end_time = i.stop
            
            # 只对整数类型的索引需要调整结束时间
            if end_time is not None and isinstance(end_time, int):
                end_time = end_time - 1
            
        else:
            raise TypeError(f"不支持的索引类型: type(i) = {type(i)}")

        # ⚠️ 整个 fetch 是「run→upload→run」的多步会话对话，必须整体持锁，
        # 否则并发线程会互相覆盖服务器端的 instruments/expressions 变量
        with DBClient.session_lock:
            df: pd.DataFrame = fetch_features_from_ddb(
                DBClient.session,
                self.instrument,
                self.field,
                start_time,
                end_time,
                self.freq
                )

        if df.empty:
            return pd.Series(dtype=float)
        
        return df