"""Run all unit tests and expanded ComfyUI contracts without a GPU or weights."""
import argparse
import pathlib
import runpy
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    comfy_root = args.comfyui_root.resolve()
    if not (comfy_root / "comfy" / "options.py").is_file():
        parser.error("--comfyui-root must point to a prepared ComfyUI checkout")
    sys.path[:0] = [str(comfy_root), str(ROOT)]
    # ComfyUI parses this on first import of its runtime. Set CPU mode before
    # loading any test module that imports h3_relay or model_management.
    sys.argv = ["h3-relay-cpu-tests", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    )
    if not result.wasSuccessful():
        return 1
    runpy.run_path(str(ROOT / "tests" / "runtime_contract.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
