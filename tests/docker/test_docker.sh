#!/usr/bin/env bash
set -xe

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
IMAGE='/data/test.tif'
OUTPUT=test-docker-output.pkl
MODEL_DIR=$DIR/../../slaid/models
rm -f $DIR/../data/$OUTPUT

for model in $(ls $MODEL_DIR); do 
  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models slaid extract_tissue.py -m /models/$model -l 0 $IMAGE /data/$OUTPUT
  ls -l $DIR/../data/$OUTPUT
  rm $DIR/../data/$OUTPUT
  
  docker run  --rm -v $DIR/../data:/data  -v $MODEL_DIR:/models slaid extract_tissue.py --only-mask -m /models/$model -l 0 $IMAGE /data/$OUTPUT
  ls -l $DIR/../data/$OUTPUT
  rm $DIR/../data/$OUTPUT

done
