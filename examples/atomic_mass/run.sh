#!/bin/bash

OUT_FILE=net_out
TRAINING=training.txt
TESTING=testing.txt

echo 'Training (please wait a few moments)...'
neurocli new -in network.txt -out $OUT_FILE
neurocli train -net $OUT_FILE -samples $TRAINING -adam default -cost mse \
   -batch 128 -stopcost 2.5 -quiet

for atomicNum in $(cat testing.txt)
do
  echo "Predicting mass for $atomicNum:"
  echo $atomicNum | neurocli run -net $OUT_FILE
done

rm net_out
