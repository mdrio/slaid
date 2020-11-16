#!/usr/bin/env bash
set -xe

pytest --spec tests
python  tests/integration/test_pipeline.py
