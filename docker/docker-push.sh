#!/usr/bin/env bash

set -x
DOCKER_ARGS=$1
while read repo; do
  ./docker_cmd.py $DOCKER_ARGS  -v $(./get_docker_tag.sh) tag  -r $repo;
  ./docker_cmd.py $DOCKER_ARGS -v $(./get_docker_tag.sh) push  -r $repo;
done < repo.txt


