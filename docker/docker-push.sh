#!/usr/bin/env bash

set -ex
TAG=$1
EXTRA_TAGS=$2
while read repo; do
  ./docker_cmd.py $DOCKER_ARGS  -v $TAG tag  -r $repo -e $EXTRA_TAGS | xargs  docker push 

done < repo.txt
