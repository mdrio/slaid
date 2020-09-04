#!/usr/bin/env bash

set -x
while read repo; do
  ./docker_cmd.py -v $(./get_docker_tag.sh) tag  -r $repo; 
  ./docker_cmd.py -v $(./get_docker_tag.sh) push  -r $repo; 
done < repo.txt


