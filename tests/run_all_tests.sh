#!/usr/bin/env bash
set -xe

python -m unittest discover -s tests -f
python  tests/integration/test_pipeline.py
