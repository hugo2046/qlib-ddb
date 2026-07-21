# tests/test_ddb_features.py
"""测试 DolphinDB 兼容函数的脚本注册顺序。

回归背景：
    ``register_ddb_functions_to_qlib`` 通过 ``Path.glob("*.dos")`` 遍历
    ``ddb_scripts/*.dos`` 并逐个 ``runFile``。``Path.glob()`` 返回顺序依赖
    文件系统 ``readdir``（macOS APFS 与 Linux ext4 顺序不同），但脚本间存在
    依赖——``ops.dos`` 定义了 ``Slope``/``Rsquare``/``Resi`` 等基础算子，
    被 ``qlib158Alpha.dos`` 跨文件引用（首次使用位于 qlib158Alpha.dos:154）。

    若 ``qlib158Alpha.dos`` 先于 ``ops.dos`` 执行，DolphinDB 服务端会抛
    ``Syntax Error: [line #154] Cannot recognize the token Slope``，导致
    qlib 初始化在 Linux 容器内失败、macOS 上“碰巧”通过。

    本测试锁定加载顺序的确定性：``ops.dos`` 必须置首，其余按文件名字母序，
    与文件系统 ``readdir`` 顺序无关。
"""

from pathlib import Path

import pytest

from qlib.data.backend.ddb_qlib.ddb_features import (
    _sort_ddb_scripts,
    register_ddb_functions_to_qlib,
)


class TestSortDdbScripts:
    """``_sort_ddb_scripts`` 是纯函数：把任意输入排成确定性顺序。"""

    @staticmethod
    def _names(scripts):
        return [Path(s).name for s in scripts]

    def test_ops_dos_always_first_regardless_of_input_order(self) -> None:
        """对抗输入（模拟 Linux ext4 把 ops.dos 排到末尾）下，ops.dos 仍置首。

        该断言与主机文件系统无关，保证在任意 readdir 顺序下都成立。
        """
        adversarial = [
            Path("qlib158Alpha.dos"),
            Path("wq101alpha.dos"),
            Path("gtja191Alpha.dos"),
            Path("ops.dos"),  # 故意放最后，模拟最坏 readdir
            Path("featureEngineering.dos"),
        ]
        result = self._names(_sort_ddb_scripts(adversarial))
        assert result[0] == "ops.dos"

    def test_remaining_scripts_in_alphabetical_order(self) -> None:
        """ops.dos 之外，其余脚本按文件名字母序加载。"""
        adversarial = [
            Path("wq101alpha.dos"),
            Path("qlib158Alpha.dos"),
            Path("ops.dos"),
            Path("gtja191Alpha.dos"),
            Path("featureEngineering.dos"),
            Path("prepareInstruments.dos"),
        ]
        result = self._names(_sort_ddb_scripts(adversarial))
        assert result == [
            "ops.dos",
            "featureEngineering.dos",
            "gtja191Alpha.dos",
            "prepareInstruments.dos",
            "qlib158Alpha.dos",
            "wq101alpha.dos",
        ]

    def test_order_is_input_independent(self) -> None:
        """确定性：打乱输入顺序，输出必须完全一致。"""
        base = [
            Path("ops.dos"),
            Path("qlib158Alpha.dos"),
            Path("wq101alpha.dos"),
            Path("featureEngineering.dos"),
        ]
        once = self._names(_sort_ddb_scripts(base))
        twice = self._names(_sort_ddb_scripts(list(reversed(base))))
        assert once == twice


class _FakeSession:
    """记录 runFile 调用顺序的假 DolphinDB 会话，避免依赖真实服务端。

    对齐 ``dolphindb.Session.runFile`` 的签名：接收一个脚本路径。
    """

    def __init__(self) -> None:
        self.files: list[str] = []

    def runFile(self, script_file) -> None:  # noqa: N802 - 对齐 DolphinDB SDK 命名
        """记录被加载的脚本文件名。"""
        self.files.append(Path(script_file).name)


@pytest.mark.usefixtures("monkeypatch")
class TestRegisterLoadsOpsFirst:
    """``register_ddb_functions_to_qlib`` 必须把 ops.dos 加载到最前。

    本机 macOS APFS 上原生 glob 顺序为
    ``[..., gtja191Alpha, wq101alpha, ops, qlib158Alpha, prepareInstruments]``
    ——ops.dos 落在 index 3。故修复前本测试在本机即失败，可直接复现 bug。
    """

    def test_register_loads_ops_dos_first(self) -> None:
        """真实函数（经 mock session）加载的第一个脚本必须是 ops.dos。"""
        session = _FakeSession()
        register_ddb_functions_to_qlib(session)  # type: ignore[arg-type]  # 鸭子类型假会话
        assert session.files[0] == "ops.dos"

    def test_register_loads_core_scripts_exactly_once(self) -> None:
        """默认仅加载核心三件套（alpha 库 119KB 改为按需惰性加载）且无重复。"""
        session = _FakeSession()
        register_ddb_functions_to_qlib(session)  # type: ignore[arg-type]  # 鸭子类型假会话
        assert session.files == [
            "ops.dos",
            "featureEngineering.dos",
            "prepareInstruments.dos",
        ]

    def test_register_preload_loads_every_script(self) -> None:
        """preload_alpha_libs=True 恢复历史全量加载行为（ops.dos 仍置首）。"""
        session = _FakeSession()
        register_ddb_functions_to_qlib(session, preload_alpha_libs=True)  # type: ignore[arg-type]
        script_dir = Path(register_ddb_functions_to_qlib.__code__.co_filename).parent / "ddb_scripts"
        expected_count = len(list(script_dir.glob("*.dos")))
        assert session.files[0] == "ops.dos"
        assert len(session.files) == expected_count
        assert len(set(session.files)) == expected_count  # 无重复


class TestLazyAlphaLibLoading:
    """alpha 因子库按字段引用惰性加载（每会话每库仅一次）。"""

    def _registered_session(self) -> _FakeSession:
        session = _FakeSession()
        register_ddb_functions_to_qlib(session)  # type: ignore[arg-type]
        session.files.clear()  # 只观察后续惰性加载
        return session

    def test_alpha_field_triggers_single_load(self) -> None:
        from qlib.data.backend.ddb_qlib.ddb_features import ensure_alpha_libs_loaded

        session = self._registered_session()
        ensure_alpha_libs_loaded(session, ["gtjaAlpha3($open,$close)", "$high"])
        assert session.files == ["gtja191Alpha.dos"]
        # 第二次引用同库：不再加载
        ensure_alpha_libs_loaded(session, ["gtjaAlpha5($close)"])
        assert session.files == ["gtja191Alpha.dos"]

    def test_plain_fields_load_nothing(self) -> None:
        from qlib.data.backend.ddb_qlib.ddb_features import ensure_alpha_libs_loaded

        session = self._registered_session()
        ensure_alpha_libs_loaded(session, ["$close", "Ref($close,1)"])
        assert session.files == []

    def test_case_insensitive_match(self) -> None:
        from qlib.data.backend.ddb_qlib.ddb_features import ensure_alpha_libs_loaded

        session = self._registered_session()
        ensure_alpha_libs_loaded(session, ["WQAlpha1($close)", "qlib158Alpha2($open)"])
        assert sorted(session.files) == ["qlib158Alpha.dos", "wq101alpha.dos"]

    def test_preloaded_session_skips_lazy_load(self) -> None:
        from qlib.data.backend.ddb_qlib.ddb_features import ensure_alpha_libs_loaded

        session = _FakeSession()
        register_ddb_functions_to_qlib(session, preload_alpha_libs=True)  # type: ignore[arg-type]
        session.files.clear()
        ensure_alpha_libs_loaded(session, ["gtjaAlpha3($open,$close)"])
        assert session.files == []
