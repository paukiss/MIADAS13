#!/usr/bin/env bash
set -euo pipefail

python -m src.cli build-master
python -m src.cli tune
python -m src.cli evaluate
python -m src.cli predict
