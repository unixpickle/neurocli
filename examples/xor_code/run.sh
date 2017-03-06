#!/bin/bash

OUT_FILE=net_out
SAMPLES=samples.txt
MESSAGE=message.txt

echo 'Training (this may take a few seconds)...'
neurocli new -in network.txt -out $OUT_FILE
neurocli train -net $OUT_FILE -samples $SAMPLES -adam default \
  -cost sigmoidce -stopcost 0.03 -step 0.01 -quiet

echo 'Decoding message as binary'
neurocli run -net $OUT_FILE -in message.txt | neurocli signbit

rm $OUT_FILE
