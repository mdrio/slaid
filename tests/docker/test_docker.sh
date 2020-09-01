#!/usr/bin/env bash
set -xe

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
OUTDIR=$DIR/../data
IMAGE='test.tif'
IMAGE_NO_EXT='test'
MODEL_DIR=$DIR/../../slaid/models

for model in $(ls $MODEL_DIR); do 
  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models  slaid  classify.py  -m /models/$model -l 0 /data/$IMAGE  -w pkl -f tissue /data
  ls -l $OUTDIR/${IMAGE_NO_EXT}.tissue.pkl
  rm $OUTDIR/${IMAGE_NO_EXT}.tissue.pkl

  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models  slaid classify.py  -m /models/$model -l 0 -w json /data/$IMAGE   -f tissue /data
  ls -l $DIR/../data/${IMAGE_NO_EXT}.tissue.json
  rm $OUTDIR/${IMAGE_NO_EXT}.tissue.json

  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models  slaid classify.py  --only-mask -m /models/$model -l 0 /data/$IMAGE  -w pkl -f tissue /data
  ls -l $DIR/../data/${IMAGE_NO_EXT}.tissue.pkl 
  rm $OUTDIR/${IMAGE_NO_EXT}.tissue.pkl 
 
  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models  slaid classify.py  --only-mask -m /models/$model  -w json -l 0 /data/$IMAGE  -f tissue /data
  ls -l $DIR/../data/${IMAGE_NO_EXT}.tissue.json
  rm $OUTDIR/${IMAGE_NO_EXT}.tissue.json
done
