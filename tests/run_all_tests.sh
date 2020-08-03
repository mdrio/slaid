#!/usr/bin/env bash
set -xe

python -m unittest
python integration/test_pipeline.py
bin/test_bin.sh
