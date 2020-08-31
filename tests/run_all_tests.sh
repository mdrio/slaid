#!/usr/bin/env bash
set -xe

python -m unittest discover -s tests
python  tests/integration/test_pipeline.py
