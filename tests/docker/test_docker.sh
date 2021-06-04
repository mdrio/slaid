#!/usr/bin/env bash
set -xe

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUTDIR=$DIR/../data
IMAGE='test.tif'
IMAGE_NO_EXT='test'
MODEL_DIR=$DIR/../../slaid/resources/models
model=tissue_model-extract_tissue_eddl_1.1.bin
for mode in serial parallel; do 
  for ext in  zarr; do 
    docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models  slaid $mode -w ${ext}  -m /models/$model -l 0 /data/$IMAGE --overwrite  -f tissue  -o/data
    ls -l $OUTDIR/${IMAGE}.${ext}
    rm -r $OUTDIR/${IMAGE}.${ext}

  done
done
