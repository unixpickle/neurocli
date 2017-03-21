# About this example

RNNs can be used to transform variable length sequences into fixed-length vectors. In neurocli, this is called a "sequence-to-vector" model. In this example, we use a sequence-to-vector model to learn a very simple checksum algorithm.

A sequence-to-vector model has two components: an RNN (defined in [rnn.txt](rnn.txt)) and an output network (defined in [out_net.txt](out_net.txt)). The RNN processes the sequence and produces data for the output network. The output network takes the RNN's output from the final time-step and produces an output vector. In a sense, you can think of the RNN as *encoding* the sequence so the output network has a vector to operate on.

Our model will take binary sequences and produce a single bit checksum. Training examples can be found in [data.txt](data.txt). In case you are wondering, the checksum is a simple XOR.

The [run.sh](run.sh) script trains a model on the data and then produces checksums for two (almost identical) sequences.
