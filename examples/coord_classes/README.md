# About this example

In this example, we are trying to classify points on an x-y plane. There are three different categories that a given point might fall under. We want to guess, for a new point, which category that point will fall under. The end goal is to be able to "color in" the entire plane by deciding which class every point should belong to. This could look like:

```
00000000000000000000000000000000000000000
00000000000000000000000000000000000000000
00000000000000000000000111111111110000000
00000000000000000000111111111111111100000
00000000000000000000111111111111111100000
00000000000000000000111111111111111100000
00000000000000000000011111111111111000000
00000000000000000000000111111111111000000
00000000000000000000000001111111100000000
00000000000000000000000000000000000000000
00000000000000000000000000000000000000000
00000000000000002222222200000000000000000
00000000000000222222222222000000000000000
00000000000002222222222222200000000000000
00000000000002222222222222220000000000000
00000000000000222222222222220000000000000
00000000000000022222222222220000000000000
00000000000000000222222222200000000000000
00000000000000000000000000000000000000000
00000000000000000000000000000000000000000
00000000000000000000000000000000000000000
```

The [points.txt](points.txt) file contains 100 points and a classification for each. Classifications are encoded as one-hot vectors, meaning that "1 0 0" indicates class 0, "0 1 0" indicates class 1, etc. We use one-hot vectors because that is what the `softmax` cost expects.

The network, as defined in [network.txt](network.txt), takes two points as input and produces three outputs. Since we use the `Softmax` layer as the last layer of the network, each output is the log-probability for the corresponding class. For example, if the network returns the following,

    -0.69315 -1.38629 -1.38629

it thinks that class 0 has an `e^-0.69 = 0.5` probability, whereas class 1 and 2 have an `e^-1.39 = 0.25` probability.

When running the network as a classifier, we want the most probable classification for every point. We can use the `neurocli max` command to select the maximum output from the softmax layer.

The [run.sh](run.sh) script trains a network, then uses the network to classify points on a grid to generate an ASCII picture like the one shown above.
