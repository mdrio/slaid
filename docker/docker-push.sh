#!/usr/bin/env bash

set -x
IMG=$1
while read repo; do  ./docker_cmd.py push -r $repo; done < repo.txt


