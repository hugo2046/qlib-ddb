"""DolphinDBDataLoader teardown 的离线回归测试。

历史 bug：``__exit__``/``__del__`` 无条件调用 ``self.session.close()``，
而 ``self.session`` 是进程级共享的 ``DBClient.session``——一个 with 块或一次 GC
就会把全局会话关掉，破坏 storage 与 feature 路径的所有其他使用方。
修复后引入 ``_owns_session`` 所有权标志：仅独占会话时才允许关闭。
"""

from qlib.data.dataset.loader import DolphinDBDataLoader


class _FakeSession:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _make_loader(owns_session: bool) -> tuple[DolphinDBDataLoader, _FakeSession]:
    """绕过构造器（构造器依赖全局 DBClient）注入假会话。"""
    loader = object.__new__(DolphinDBDataLoader)
    session = _FakeSession()
    loader.session = session
    loader._owns_session = owns_session
    return loader, session


def test_shared_session_never_closed():
    """共享会话（默认）在 __exit__ 与 __del__ 中都不得被关闭。"""
    loader, session = _make_loader(owns_session=False)
    loader.__exit__(None, None, None)
    loader.__del__()
    assert session.close_calls == 0, "回归：共享的全局 session 被关闭"


def test_owned_session_closed_on_exit():
    """独占会话在 __exit__ 时正常关闭。"""
    loader, session = _make_loader(owns_session=True)
    loader.__exit__(None, None, None)
    assert session.close_calls == 1


def test_del_without_attributes_is_safe():
    """构造中途失败（属性未设置）时 __del__ 不应抛出。"""
    loader = object.__new__(DolphinDBDataLoader)
    loader.__del__()  # 不应抛出
