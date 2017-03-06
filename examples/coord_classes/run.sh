#!/bin/bash

OUT_FILE=net_out
POINTS=points.txt

echo 'Training (this may take a few seconds)...'
neurocli new -in network.txt -out $OUT_FILE
neurocli train -net $OUT_FILE -samples $POINTS -adam default \
  -cost softmax -batch 10 -stopcost 0.01 -quiet

echo 'Coloring in grid'
for row in {0..20}
do
  scaledRow=$(echo "scale=2; $row/20" | bc)
  for col in {0..40}
  do
    scaledCol=$(echo "scale=3; $col/40" | bc)
    out=$(echo "$scaledCol $scaledRow" | neurocli run -net $OUT_FILE | neurocli max)
    echo -n $out
  done
  echo
done

rm $OUT_FILE
