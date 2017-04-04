#!/bin/bash

OUT_FILE=net_out
POINTS=points.txt

echo 'Training (this may take a few seconds)...'
neurocli new -in network.txt -out $OUT_FILE
neurocli train -net $OUT_FILE -samples $POINTS -adam default \
  -cost softmax -batch 10 -stopcost 0.01 -quiet

echo 'Coloring in grid'
for row in $(seq 0 0.05 1)
do
  for col in $(seq 0 0.025 1)
  do
    out=$(echo "$col $row" | neurocli run -net $OUT_FILE | neurocli max)
    echo -n $out
  done
  echo
done

rm $OUT_FILE
