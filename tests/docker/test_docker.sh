#!/usr/bin/env bash
set -xe

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUTDIR=$DIR/../data
IMAGE='test.tif'
IMAGE_NO_EXT='test'
MODEL_DIR=$DIR/../../slaid/resources/models
model=extract_tissue_eddl-1.0.0.bin
for mode in serial parallel; do 
  for ext in tiledb zarr; do 
    docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models  slaid  classify.py $mode -w ${ext}  -m /models/$model -l 0 /data/$IMAGE --overwrite  -f tissue /data
    ls -l $OUTDIR/${IMAGE}.${ext}
    rm -r $OUTDIR/${IMAGE}.${ext}


    docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models  slaid classify.py $mode -w ${ext} --patch-size 256x256 -m /models/$model -l 0 /data/$IMAGE  --overwrite  -f tissue /data
    ls -l $DIR/../data/${IMAGE}.${ext}
    rm -r $OUTDIR/${IMAGE}.${ext}
 
  done
done
