#!/usr/bin/env bash
version=$(cat ../VERSION)
git_branch=$(git rev-parse --abbrev-ref HEAD | tr /: _)
if [ $git_branch == 'master' ]; then
  echo $version
else
  echo $version-$git_branch
fi
