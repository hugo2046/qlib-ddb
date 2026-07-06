# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""qlib.utils.paral.ParallelExt 回归测试。

joblib 1.3 起 ``Parallel`` 的 ``_backend_args`` 属性改名为 ``_backend_kwargs``，
而 ``ParallelExt.__init__`` 仍写 ``self._backend_args``，在 multiprocessing 后端下
抛 ``AttributeError: 'ParallelExt' object has no attribute '_backend_args'``，
导致任何走并行数据计算的路径（如 ``D.features`` → ``dataset_processor``）崩溃。
"""

import unittest

from joblib import Parallel, delayed

from qlib.utils.paral import ParallelExt


def _backend_kwargs_dict(parallel: Parallel):
    """取 Parallel 实例上的 backend-kwargs 字典（兼容新旧 joblib 命名）。"""
    return getattr(parallel, "_backend_kwargs", None) or getattr(parallel, "_backend_args", None)


def _double(x: int) -> int:
    """模块级函数：multiprocessing spawn 后端要求可 pickle，故不能用 lambda。"""
    return x * 2


class TestParallelExt(unittest.TestCase):
    def test_maxtasksperchild_multiprocessing_no_raise(self):
        """multiprocessing 后端构造 ParallelExt 不应抛 AttributeError，
        且 maxtasksperchild 应注入到 backend kwargs（供 configure 时传给 Pool）。"""
        pe = ParallelExt(n_jobs=2, backend="multiprocessing", maxtasksperchild=2)
        bk = _backend_kwargs_dict(pe)
        self.assertIsNotNone(bk, "Parallel 实例缺少 _backend_kwargs/_backend_args")
        self.assertEqual(bk.get("maxtasksperchild"), 2)

    def test_maxtasksperchild_multiprocessing_dispatch(self):
        """端到端：multiprocessing 后端实际派发任务不报错，
        验证 maxtasksperchild 注入到 Pool 时不会破坏池创建。"""
        pe = ParallelExt(n_jobs=2, backend="multiprocessing", maxtasksperchild=1)
        res = pe(delayed(_double)(i) for i in range(4))
        self.assertEqual(sorted(res), [0, 2, 4, 6])


if __name__ == "__main__":
    unittest.main()
