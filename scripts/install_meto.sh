#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f "meto/setup.py" ] || [ ! -f "meto/meto/__init__.py" ] || [ ! -f "meto/src/bindings.cpp" ]; then
  echo "[ERROR] ./meto is missing or incomplete."
  echo "Copy NVlabs/EdgeRunner/meto to this project root so the layout is:"
  echo "  ./meto/setup.py"
  echo "  ./meto/meto/__init__.py"
  echo "  ./meto/src/bindings.cpp"
  exit 1
fi

python -m pip uninstall -y meto >/dev/null 2>&1 || true
python -m pip install -U pip wheel ninja packaging "setuptools==69.5.1"
python -m pip install "pybind11==2.11.1" numpy trimesh kiui pymeshlab
python -m pip install --no-build-isolation -e ./meto -v

# Important: ./meto must be discoverable before the repository-root namespace
# directory named "meto". The training code also handles this internally, but
# this export makes standalone tests deterministic.
export PYTHONPATH="$PWD/meto:$PWD:${PYTHONPATH:-}"
python scripts/diagnose_meto.py
