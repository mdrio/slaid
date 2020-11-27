#!/usr/bin/env bash
set -xe

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUTDIR=$DIR/../data
IMAGE='test.tif'
IMAGE_NO_EXT='test'
MODEL_DIR=$DIR/../../slaid/resources/models
model=extract_tissue_eddl-1.0.0.bin
for mode in serial parallel; do 
  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models  slaid  classify.py $mode  -m /models/$model -l 0 /data/$IMAGE --overwrite  -f tissue /data
  ls -l $OUTDIR/${IMAGE}.zarr
  rm -r $OUTDIR/${IMAGE}.zarr


  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models  slaid classify.py $mode --patch-size 256x256 -m /models/$model -l 0 /data/$IMAGE  --overwrite  -f tissue /data
  ls -l $DIR/../data/${IMAGE}.zarr 
  rm -r $OUTDIR/${IMAGE}.zarr 
 
done
