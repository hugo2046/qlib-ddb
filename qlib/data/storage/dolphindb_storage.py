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
from qlib.data.data import DolphinDB
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

    def query(self, expr):
        df = pd.DataFrame()

        try:
            df = DolphinDB.run(expr)
        except Exception as e:
            logger.error(e)

        return df

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
    def uri(self):
        return self.table_name

    def exists(self, table_name):

        storage_name, table_name = table_name.split(".")
        return DolphinDB.run(f"""existsTable("dfs://{storage_name}","{table_name}")""")

    def check(self):
        """check self.uri

        Raises
        -------
        ValueError
        """
        if not self.exists(self.uri):
            raise ValueError(f"{self.storage_name} not exists: {self.uri}")


class DBCalendarStorage(DBStorageMixin, CalendarStorage):

    def __init__(self, freq: str, future: bool, provider_uri: dict = None, **kwargs):
        super(DBCalendarStorage, self).__init__(freq, future, **kwargs)
        self.future = future
        self.enable_read_cache = True  # TODO: make it configurable
        self.region = C["region"]

        # calendars.day
        self.table_name = f"{self.storage_name}s.{self._freq_db}"
        self.ddb_table = (
            f"""loadTable("dfs://Qlib{self.storage_name.title()}s","{self._freq_db}")"""
        )

    # 从这里获取数据
    def _read_calendar(self) -> List[CalVT]:
        # NOTE:
        # if we want to accelerate partial reading calendar
        # we can add parameters like `skip_rows: int = 0, n_rows: int = None` to the interface.
        # Currently, it is not supported for the txt-based calendar

        if not self.exists(self.table_name):
            self._write_calendar(values=[])

        sql = f"""SELECT * FROM {self.ddb_table}"""
        df = DolphinDB.run(sql)
        if df.empty:
            return []

        return df["TRADE_DAYS"].tolist()

    def _write_calendar(self, values: Iterable[CalVT], mode: str = "wb"):
        pass

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

    @property
    def data(self) -> List[CalVT]:
        # self.check()
        # If cache is enabled, then return cache directly
        if self.enable_read_cache:
            key = "orig_file" + str(self.uri)
            if key not in H["c"]:
                H["c"][key] = self._read_calendar()
            _calendar = H["c"][key]
        else:
            _calendar = self._read_calendar()
        if Freq(self._freq_db) != Freq(self.freq):
            _calendar = resam_calendar(
                np.array(list(map(pd.Timestamp, _calendar))),
                self._freq_db,
                self.freq,
                self.region,
            )
        return _calendar

    def _get_storage_freq(self) -> List[str]:
        return sorted(
            set(map(lambda x: x.stem.split("_")[0], self.uri.parent.glob("*.txt")))
        )

    def extend(self, values: Iterable[CalVT]) -> None:
        self._write_calendar(values, mode="ab")

    def clear(self) -> None:
        self._write_calendar(values=[])

    def index(self, value: CalVT) -> int:
        self.check()
        calendar = self._read_calendar()
        return int(np.argwhere(calendar == value)[0])

    def insert(self, index: int, value: CalVT):
        calendar = self._read_calendar()
        calendar = np.insert(calendar, index, value)
        self._write_calendar(values=calendar)

    def remove(self, value: CalVT) -> None:
        self.check()
        index = self.index(value)
        calendar = self._read_calendar()
        calendar = np.delete(calendar, index)
        self._write_calendar(values=calendar)

    def __setitem__(
        self, i: Union[int, slice], values: Union[CalVT, Iterable[CalVT]]
    ) -> None:
        calendar = self._read_calendar()
        calendar[i] = values
        self._write_calendar(values=calendar)

    def __delitem__(self, i: Union[int, slice]) -> None:
        self.check()
        calendar = self._read_calendar()
        calendar = np.delete(calendar, i)
        self._write_calendar(values=calendar)

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, List[CalVT]]:
        self.check()
        return self._read_calendar()[i]

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
        self.table_name = f"Qlib{self.storage_name.title()}s.{market.lower()}"
        self.ddb_table = f"""loadTable("dfs://Qlib{self.storage_name.title()}s","{market.lower()}")"""

    def _read_instrument(self) -> Dict[InstKT, InstVT]:
        if not self.exists(self.table_name):
            raise FileNotFoundError(self.table_name)

        _instruments = dict()

        sql = f"""SELECT * FROM {self.ddb_table}"""

        df = DolphinDB.run(sql)
        for row in df.itertuples(index=False):
            _instruments.setdefault(row[0], []).append((row[1], row[2]))

        return _instruments

    def _write_instrument(self, data: Dict[InstKT, InstVT] = None) -> None:
        raise NotImplementedError(f"Please use other database tools to write!")

    def clear(self) -> None:
        self._write_instrument(data={})

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        self.check()
        return self._read_instrument()

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        inst = self._read_instrument()
        inst[k] = v
        self._write_instrument(inst)

    def __delitem__(self, k: InstKT) -> None:
        self.check()
        inst = self._read_instrument()
        del inst[k]
        self._write_instrument(inst)

    def __getitem__(self, k: InstKT) -> InstVT:
        self.check()
        return self._read_instrument()[k]

    def update(self, *args, **kwargs) -> None:
        if len(args) > 1:
            raise TypeError(f"update expected at most 1 arguments, got {len(args)}")
        inst = self._read_instrument()
        if args:
            other = args[0]  # type: dict
            if isinstance(other, Mapping):
                for key in other:
                    inst[key] = other[key]
            elif hasattr(other, "keys"):
                for key in other.keys():
                    inst[key] = other[key]
            else:
                for key, value in other:
                    inst[key] = value
        for key, value in kwargs.items():
            inst[key] = value

        self._write_instrument(inst)

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
        self.instrument = (
            instrument.upper()
            if isinstance(instrument, str)
            else list(map(lambda x: x.upper(), instrument))
        )
        self.db_path = f"dfs://Qlib{self.storage_name.title()}s{self.freq.title()}"
        self.table_name = "Features"

        self.calendar = self._read_calendar()

        self.has_field = self._exists_table() and self.field_exists()
        self.storage_start_index = self.start_index
        self.storage_end_index = self.end_index

    def _exists_table(self) -> bool:

        return DolphinDB.existsTable(self.db_path, self.table_name)

    def _read_calendar(self) -> List[CalVT]:
        # NOTE:
        # if we want to accelerate partial reading calendar
        # we can add parameters like `skip_rows: int = 0, n_rows: int = None` to the interface.
        # Currently, it is not supported for the txt-based calendar

        if not DolphinDB.existsTable("dfs://QlibCalendars", self.freq):
            raise NotImplementedError(
                f"Table dfs://QlibCalendars/{self.freq} not exists."
            )

        sql: str = f"""SELECT * FROM loadTable("dfs://QlibCalendars","{self.freq}")"""
        df: pd.DataFrame = DolphinDB.run(sql)

        # check the type of the query result
        if not isinstance(df, (pd.DataFrame, pd.Series)):
            raise TypeError(f"查询结果类型不是DataFrame或Series.[sql expr:{sql}]")

        if df.empty:
            return []

        return df["TRADE_DAYS"].tolist()

    def field_exists(self):

        ddb_fields = (
            DolphinDB.loadTable(self.table_name, self.db_path).schema["name"].tolist()
        )

        if self.field not in ddb_fields:
            raise KeyError(f"field {self.field} not exists")

        df = (
            DolphinDB.loadTable(self.table_name, self.db_path)
            .select(f"count({self.field})")
            .where(f"code=='{self.instrument}'")
            .toDF()
        )
        if df.empty:
            return False
        return df.iloc[0, 0] != 0

    def clear(self):
        with self.uri.open("wb") as _:
            pass

    @property
    def data(self) -> pd.Series:
        return self[:]

    def write(self, data_array: Union[List, np.ndarray], index: int = None) -> None:
        raise NotImplementedError(f"Please use other database tools to write!")

    @property
    def start_index(self) -> Union[int, None]:
        if not self.has_field:
            return None

        df = (
            DolphinDB.loadTable(self.table_name, self.db_path)
            .select("date(min(date))")
            .where(f"code=='{self.instrument}'")
            .toDF()
        )
        if df.empty:
            return None
        else:
            return self.calendar.index(df.iat[0, 0])

    @property
    def end_index(self) -> Union[int, None]:
        if not self.has_field:
            return None

        return self.start_index + len(self) - 1

    def __getitem__(self, i: Union[int, slice]) -> Union[Tuple[int, float], pd.Series]:

        if not self.has_field:
            if isinstance(i, int):
                return None, None
            elif isinstance(i, slice):
                return pd.Series(dtype=np.float32)
            else:
                raise TypeError(f"type(i) = {type(i)}")

        if isinstance(i, int):

            if self.storage_start_index > i:
                raise IndexError(f"{i}: start index is {self.storage_start_index}")

            watch_dt = self.calendar[i - self.storage_start_index + 1].strftime(
                "%Y.%m.%d"
            )

            df = (
                DolphinDB.loadTable(self.table_name, self.db_path)
                .select(self.field)
                .where(f"code=='{self.instrument}' and date=='{watch_dt}'")
                .toDF()
            )

            if df.empty:
                return i, None
            else:
                return i, df.iloc[0][self.field]

        elif isinstance(i, slice):
            start_index = self.storage_start_index if i.start is None else i.start
            end_index = self.storage_end_index if i.stop is None else i.stop - 1
            si = max(start_index, self.storage_start_index)
            if si > end_index:
                return pd.Series(dtype=np.float32)

            # start_id = si - self.storage_start_index + 1
            # end_id = end_index - self.storage_start_index + 1
            start_dt = self.calendar[si].strftime("%Y.%m.%d")
            end_dt = self.calendar[end_index].strftime("%Y.%m.%d")

            df = (
                DolphinDB.loadTable(self.table_name, self.db_path)
                .select(self.field)
                .where(
                    f"code=='{self.instrument}' and date between pair({start_dt},{end_dt})"
                )
                .toDF()
            )
            data = df[self.field].to_list()
            # print(si, self.storage_start_index, self.storage_end_index,count)
            series = pd.Series(data, index=pd.RangeIndex(si, si + len(data)))
            # print(series)
            return series
        else:
            raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:

        df = (
            DolphinDB.loadTable(self.table_name, self.db_path)
            .select("count(*)")
            .where(f"code=='{self.instrument}'")
            .toDF()
        )
        if df.empty:
            return None
        else:
            return df.iloc[0, 0] - 1



