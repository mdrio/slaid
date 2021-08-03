#!/usr/bin/env bash

set -x
TAG=$1
DOCKER_ARGS=$2
while read repo; do
  ./docker_cmd.py $DOCKER_ARGS  -v $TAG tag  -r $repo;
  ./docker_cmd.py $DOCKER_ARGS -v $TAG push  -r $repo;
done < repo.txt


