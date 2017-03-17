# Bidirectional XOR

This is a minimal demonstration of bidirectional RNNs. In a traditional RNN, the output at time-step *t* can only depend on inputs from time-steps before or at *t*. In a bidirectional RNN, every output can depend on every input. This makes it possible to do things like the bidirectional XOR, as I do in this example.

In the bidirectional XOR problem, at output *t* we output `in[t-1] ^ in[t+1]` where `^` is XOR. If `t-1` or `t+1` goes out of bounds, we treat that input as a 0. For example, the sequence `1 0 0` gets turned into `0 1 0`.

# Bidirectional RNNs

To create a bidirectional RNN, we must define three components:

 * Forward RNN: an RNN which runs forwards on the input sequence, like a normal RNN.
 * Backward RNN: an RNN which runs backwards on the input sequence.
 * Mixer: a feed-forward neural network that combines outputs from the forward and backward RNNs.

The forward and backward RNNs both operate on the input sequence, just in reverse orders. As such, they share the same input size (1 in this case). However, they needn't have the same output size. The output sizes are a tunable hyper-parameter, similar to the size of a hidden layer in a traditional neural network.

The outputs of the forward and backward RNNs are concatenated before being fed to the mixer, meaning that the mixer's input size should be the sum of the two RNNs' output sizes. The mixer uses its input to produce the bidirectional RNN's final output at every time-step.

# Model setup

In this case, the forward and backward RNNs have an input size of 1. I chose to give them both an output size of 5. Since I made the forward and backward RNNs identical, I use [fwd_back.txt](fwd_back.txt) to create both of them.

The mixer, defined in [mixer.txt](mixer.txt), has an input of size 10 (i.e. 5+5) and an output of size 1 (one output bit). Since the outputs are bits, I use the `sigmoice` cost and do not use an explicit output activation function in the mixer.

# Evaluation

The [run.sh](run.sh) script creates and trains the bidirectional RNN. It then evaluates the RNN on several inputs. The output should look like:

```
Creating network...
Training (this may take a few seconds)...
Producing output for 1 0 0:
0 1 0
Producing output for 0 1 0 0 1 1 0 1 1 1:
1 0 1 1 1 1 0 1 0 1
```
