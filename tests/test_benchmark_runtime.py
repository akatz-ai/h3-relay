from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark/run_saved_workflow_benchmark.py"


def benchmark_module():
    spec = importlib.util.spec_from_file_location("h3_relay_benchmark", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkRuntimeTest(unittest.TestCase):
    def test_cgroup_memory_for_pid_resolves_unified_cgroup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_root = root / "proc"
            cgroup_root = root / "cgroup"
            (proc_root / "123").mkdir(parents=True)
            (proc_root / "123" / "cgroup").write_text(
                "0::/user.slice/test.scope\n",
                encoding="utf-8",
            )
            memory_file = cgroup_root / "user.slice" / "test.scope" / "memory.current"
            memory_file.parent.mkdir(parents=True)
            memory_file.write_text("42\n", encoding="utf-8")

            resolved = benchmark_module().cgroup_memory_for_pid(
                123,
                proc_root=proc_root,
                cgroup_root=cgroup_root,
            )

            self.assertEqual(resolved, memory_file)

    def test_service_details_accepts_standalone_process_pid(self) -> None:
        module = benchmark_module()
        expected = Path("/sys/fs/cgroup/test/memory.current")
        with mock.patch.object(module, "cgroup_memory_for_pid", return_value=expected):
            self.assertEqual(module.service_details(456), (456, expected))
