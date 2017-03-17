#!/bin/bash

OUT_FILE=net_out
SAMPLES=data.txt

echo 'Creating network...'
neurocli new -in fwd_back.txt -out forward
neurocli new -in fwd_back.txt -out backward
neurocli new -in mixer.txt -out mixer
neurocli bidir -forward forward -backward backward -mixer mixer -out $OUT_FILE
rm forward backward mixer

echo 'Training (this may take a few seconds)...'
neurocli train -adam default -samples $SAMPLES -net $OUT_FILE \
  -cost sigmoidce -batch 8 -stopcost 0.05 -quiet

echo 'Producing output for 1 0 0:'
echo '1 0 0' | neurocli run -net $OUT_FILE | neurocli signbit

echo 'Producing output for 0 1 0 0 1 1 0 1 1 1:'
echo '0 1 0 0 1 1 0 1 1 1' | neurocli run -net $OUT_FILE | neurocli signbit

rm $OUT_FILE
