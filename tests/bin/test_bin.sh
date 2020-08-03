#!/usr/bin/env bash
set -xe

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
get_tissue_mask.py -l 0 $DIR/../data/input.tiff /tmp/output_get_tissue_mask.pkl
ls -l /tmp/output_get_tissue_mask.pkl


extract_tissue.py -l 0 $DIR/../data/input.tiff /tmp/output_extract_tissue.pkl
ls -l /tmp/output_extract_tissue.pkl
