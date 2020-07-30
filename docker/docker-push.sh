#!/usr/bin/env bash

IMG=$1
while read repo; do docker tag $IMG $repo/$IMG; docker push  $repo/$IMG; done < repo.txt


