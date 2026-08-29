#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

mode="full"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
  mode="$1"
  shift
fi

case "$mode" in
  preflight|smoke|single|full) ;;
  *)
    echo "Usage: ./reproduce.sh [preflight|smoke|single|full] [options]" >&2
    exit 2
    ;;
esac

version_probe='import sys, torch, numpy, scipy, tqdm; raise SystemExit(0 if (sys.version_info[:2] == (3, 8) and torch.__version__.split("+", 1)[0] == "1.13.1" and numpy.__version__ == "1.21.6" and scipy.__version__ == "1.7.3" and tqdm.__version__ == "4.67.1") else 1)'

python_is_exact() {
  "$1" -c "$version_probe" >/dev/null 2>&1
}

conda_environment_is_exact() {
  conda run -n "$1" python -c "$version_probe" >/dev/null 2>&1
}

run_with_python() {
  local python_path=$1
  shift
  exec "$python_path" src/scripts/reproduce_tcf.py "$mode" "$@"
}

if [ -n "${TCF_PYTHON:-}" ]; then
  if [ ! -x "$TCF_PYTHON" ]; then
    echo "TCF_PYTHON is not executable: $TCF_PYTHON" >&2
    exit 2
  fi
  run_with_python "$TCF_PYTHON" "$@"
fi

if command -v python >/dev/null 2>&1 && python_is_exact "$(command -v python)"; then
  run_with_python "$(command -v python)" "$@"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "No exact Python environment was found and conda is unavailable." >&2
  echo "Install Miniconda, or set TCF_PYTHON to a compatible interpreter." >&2
  exit 2
fi

use_cpu=false
if [ "${TCF_DEVICE:-}" = "cpu" ]; then
  use_cpu=true
fi
for argument in "$@"; do
  if [ "$argument" = "--no-cuda" ]; then
    use_cpu=true
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  use_cpu=true
fi

if [ "$use_cpu" = true ]; then
  environment_name="tcf-cpu"
  environment_file="tcf_env_cpu.yaml"
  device_arguments=(--no-cuda)
else
  environment_name="tcf"
  environment_file="tcf_env.yaml"
  device_arguments=()
fi

if ! conda run -n "$environment_name" python -V >/dev/null 2>&1; then
  echo "[setup] Creating conda environment $environment_name from $environment_file"
  conda env create -f "$environment_file"
elif ! conda_environment_is_exact "$environment_name"; then
  echo "[setup] Updating conda environment $environment_name from $environment_file"
  conda env update -n "$environment_name" -f "$environment_file" --prune
fi

if ! conda_environment_is_exact "$environment_name"; then
  echo "Conda environment $environment_name does not match $environment_file." >&2
  exit 2
fi

exec conda run --no-capture-output -n "$environment_name" \
  python src/scripts/reproduce_tcf.py "$mode" "${device_arguments[@]}" "$@"
