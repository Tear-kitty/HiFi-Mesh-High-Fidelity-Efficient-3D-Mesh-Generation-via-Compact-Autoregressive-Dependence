#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
local_meto = root / "meto"

print("cwd:", os.getcwd())
print("project root:", root)
print("python:", sys.executable)
print("sys.path first 8:")
for i, p in enumerate(sys.path[:8]):
    print(f"  {i}: {p!r}")

print("\nexpected files:")
for rel in ["meto/setup.py", "meto/meto/__init__.py", "meto/src/bindings.cpp"]:
    print(f"  {rel}:", (root / rel).exists())

print("\nfind_spec before path fix:")
spec = importlib.util.find_spec("meto")
print("  spec:", spec)
print("  origin:", getattr(spec, "origin", None))
print("  locations:", list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else None)

if (local_meto / "meto" / "__init__.py").exists():
    sys.path.insert(0, str(local_meto))
for name in list(sys.modules):
    if name == "meto" or name.startswith("meto."):
        del sys.modules[name]

print("\nfind_spec after path fix:")
spec = importlib.util.find_spec("meto")
print("  spec:", spec)
print("  origin:", getattr(spec, "origin", None))
print("  locations:", list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else None)

print("\nimport test:")
try:
    meto = importlib.import_module("meto")
    print("  meto module:", meto)
    print("  meto.__file__:", getattr(meto, "__file__", None))
    print("  has Engine:", hasattr(meto, "Engine"))
    if hasattr(meto, "Engine"):
        e = meto.Engine(discrete_bins=512, backend="LR_ABSCO")
        print("  Engine OK, num_tokens:", e.num_tokens)
except Exception as exc:
    print("  FAILED:", type(exc).__name__, exc)
    raise
