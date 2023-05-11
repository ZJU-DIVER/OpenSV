#!/bin/sh
DIR="./phoneme"
mkdir $DIR
cd $DIR

rm -rf php8Mz7BG.arff
wget --content-disposition https://www.openml.org/data/download/1592281/php8Mz7BG
mv php8Mz7BG.arff phoneme.arff
cd ..