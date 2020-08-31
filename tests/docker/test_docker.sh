#!/usr/bin/env bash
set -xe

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
IMAGE='test.tif'
OUTPUT=test-docker-output.pkl
MODEL_DIR=$DIR/../../slaid/models
rm -f $DIR/../data/$OUTPUT

for model in $(ls $MODEL_DIR); do 
  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models slaid classify.py -m /models/$model -l 0 /data/$IMAGE  -w pkl
  ls -l $DIR/../data/${IMAGE}.output.pkl
  rm $DIR/../data/${IMAGE}.output.pkl

  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models slaid classify.py -m /models/$model -l 0 /data/$IMAGE
  ls -l $DIR/../data/${IMAGE}.output.json
  rm $DIR/../data/${IMAGE}.output.json
  
  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models slaid classify.py --only-mask -m /models/$model -l 0 /data/$IMAGE  -w pkl
  ls -l $DIR/../data/${IMAGE}.output.pkl
  rm $DIR/../data/${IMAGE}.output.pkl

  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models slaid classify.py --only-mask -m /models/$model -l 0 /data/$IMAGE 
  ls -l $DIR/../data/${IMAGE}.output.json
  rm $DIR/../data/${IMAGE}.output.json
done
