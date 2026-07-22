"""OPERATOR_MAPPING 的守卫测试。

历史问题：dict 字面量中 ``"Slope"``/``"Resi"`` 出现过两次（先映射到 DDB 内置的
``mslr``/``mmse``，后映射到 ops.dos 自定义的 ``Slope``/``Resi``），后者静默覆盖
前者——生效行为一直是 ops.dos 版本。清理后本测试锁定：
1. 生效映射不变（ops.dos 自定义算子）；
2. dict 字面量中不再允许出现重复键（重复键会被 Python 静默覆盖，难以察觉）。
"""

import ast
import inspect

import qlib.data.backend.ddb_qlib.ddb_features as ddb_features_mod
from qlib.data.backend.ddb_qlib.ddb_features import OPERATOR_MAPPING


def test_ops_dos_custom_operators_are_live():
    """ops.dos 自定义算子必须是生效映射（曾被前置重复键遮蔽混淆）。"""
    assert OPERATOR_MAPPING["Slope"] == "Slope"
    assert OPERATOR_MAPPING["Resi"] == "Resi"
    assert OPERATOR_MAPPING["Rsquare"] == "Rsquare"
    # 抽查若干常规映射未被误删
    assert OPERATOR_MAPPING["Ref"] == "move"
    assert OPERATOR_MAPPING["Mean"] == "mavg"
    assert OPERATOR_MAPPING["Std"] == "mstd"


def test_no_duplicate_keys_in_mapping_literal():
    """dict 字面量禁止重复键（后者静默覆盖前者，属隐蔽 bug 温床）。"""
    tree = ast.parse(inspect.getsource(ddb_features_mod))
    for node in ast.walk(tree):
        # OPERATOR_MAPPING: Dict = {...} 是 AnnAssign；兼容无注解的 Assign 写法
        if isinstance(node, ast.AnnAssign):
            target_id = getattr(node.target, "id", "")
        elif isinstance(node, ast.Assign):
            target_id = getattr(node.targets[0], "id", "")
        else:
            continue
        if target_id == "OPERATOR_MAPPING" and isinstance(node.value, ast.Dict):
            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
            dupes = {k for k in keys if keys.count(k) > 1}
            assert not dupes, f"OPERATOR_MAPPING 存在重复键: {dupes}"
            return
    raise AssertionError("未找到 OPERATOR_MAPPING 字面量定义")
