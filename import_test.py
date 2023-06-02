import traceback
import types
import sys

T2T_PREFIX = "tensor2tensor."
TOTAL_NUM_IMPORTS = 106


def global_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__


def module_imports(prefix: str):
    return [
        key for key in
        sys.modules.keys()
        if not prefix or key.startswith(prefix)
    ]


def test_import(command: str, prefix: str, total_num_modules: int):
    try:
        exec(command)
    except Exception:
        imports = sorted(list(module_imports(prefix)))
        pct = len(imports) / total_num_modules * 100
        traceback.print_exc()
        print(f"Imported {len(imports)}/{total_num_modules} ({pct}%) modules")
        print(imports)
    else:
        print("Success!")


test_import(
    command="from tensor2tensor.bin import t2t_trainer",
    prefix=T2T_PREFIX,
    total_num_modules=TOTAL_NUM_IMPORTS
)