from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_METO = ROOT / "meto"

print("python:", sys.executable)
print("cwd:", os.getcwd())
print("project root:", ROOT)
print("local meto exists:", LOCAL_METO.exists())
print("expected files:")
for rel in ["setup.py", "meto/__init__.py", "src/bindings.cpp", "include"]:
    p = LOCAL_METO / rel
    print(f"  {rel}: {p.exists()} -> {p}")

if (LOCAL_METO / "meto" / "__init__.py").exists():
    sys.path.insert(0, str(LOCAL_METO))

for name in list(sys.modules):
    if name == "meto" or name.startswith("meto."):
        del sys.modules[name]

try:
    meto = importlib.import_module("meto")
    print("meto module:", meto)
    print("meto.__file__:", getattr(meto, "__file__", None))
    print("meto.__path__:", list(getattr(meto, "__path__", [])))
    from meto import Engine
    e = Engine(discrete_bins=512, backend="LR_ABSCO")
    print("Engine OK")
    print("num_tokens:", e.num_tokens)
except Exception as exc:
    print("FAILED:", type(exc).__name__, exc)
    print("first 10 sys.path entries:")
    for item in sys.path[:10]:
        print(" ", item)
    raise
