#!/usr/bin/env bash
set -xe

pytest --spec -x tests
python  tests/integration/test_pipeline.py
