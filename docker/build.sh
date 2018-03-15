#!/bin/bash

mkdir -p dist

echo "Building docker image..."
#mkdir -p src
#rsync -av ../agents  ../data ../gym-tigrillo ../notebooks ../scripts src
docker stop fabo-thesis ; docker rm fabo-thesis
docker build -t fabo-thesis .

#echo "Saving docker image..."
#docker save tigrillo-rl | gzip > dist/rvai_image.tar.gz
