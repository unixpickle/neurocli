#!/bin/bash

TEMP=tmp
mkdir $TEMP
trap "rm -r $TEMP" EXIT

echo 'Creating network...'
neurocli new -in rnn.txt -out $TEMP/rnn
neurocli new -in out_net.txt -out $TEMP/out_net
neurocli seq2vec -rnn $TEMP/rnn -outnet $TEMP/out_net -out $TEMP/network

echo 'Training network (this may take a minute)...'
neurocli train -adam default -samples data.txt -net $TEMP/network \
  -cost sigmoidce -batch 64 -stopcost 0.5 -step 0.01 -quiet
neurocli train -adam default -samples data.txt -net $TEMP/network \
  -cost sigmoidce -batch 64 -stopcost 0.1 -step 0.0003 -quiet

MESSAGE1="0 1 0 0 1 0 0 0 0 1 1 0 1 0 0 1 1 1 1 1 0 1 0 1 1 1 0 1 0 0 1 0 0 0 1 0 0 0 0 1"
MESSAGE2="0 1 0 0 1 0 0 0 0 1 1 1 1 0 0 1 1 1 1 1 0 1 0 1 1 1 0 1 0 0 1 0 0 0 1 0 0 0 0 1"
echo -n 'Message checksum: '
echo "$MESSAGE1" | neurocli run -net $TEMP/network | neurocli signbit
echo -n 'Corrupted checksum: '
echo "$MESSAGE2" | neurocli run -net $TEMP/network | neurocli signbit
