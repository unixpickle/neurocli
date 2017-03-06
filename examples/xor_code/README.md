# About this example

In this example, we are trying to crack a code. The code takes a string of bits (e.g. "1 0 1") and decodes them to another string (e.g. "0 1 1"). We are given a lot of examples of strings and their decodings in [training.txt](training.txt). We want a neural net to learn the code, then use its knowledge to decode the contents of [message.txt](message.txt).

Since the messages can be of varying length, we will use a recurrent neural network to solve this problem. An LSTM with several hidden units will do the job (see [network.txt](network.txt) for details). The network takes in a bit and produces a bit at every time-step.

The [run.sh](run.sh) script trains a network, then evaluates it on the message. It produces the following output:

```
Training (this may take a few seconds)...
Decoding message as binary
0 1 1 0 1 0 0 0 0 1 1 0 0 1 0 1 0 1 1 0 1 1 0 0 0 1 1 0 1 1 0 0 0 1 1 0 1 1 1 1 0 0 1 0 0 0 0 1
```

By pasting that binary string into our favorite binary-to-text converter, we get the message "hello!".

# So, what really was the code?

The code is a simple XOR mask. Every triplet of bits is XOR'd with the mask "1 1 0".
