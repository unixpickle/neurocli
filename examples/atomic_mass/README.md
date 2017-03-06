# About this demo

This demo trains a neural network to predict an element's average atomic mass, given its atomic number.

The training data, located in [training.txt](training.txt), maps atomic numbers to masses. There are a few missing elements; these are listed in [testing.txt](testing.txt).

The [run.sh](run.sh) script uses the neurocli command to train a small neural network on the training data. This takes about 5 seconds. It then goes through the testing elements and uses the network to predict their atomic masses. Here are the results:

| Element    | Prediction          | Actual |
|------------|---------------------|--------|
| 8          | 16.56               | 16.00  |
| 32         | 72.31               | 72.64  |
| 45         | 103.37              | 102.91 |
| 55         | 133.14              | 132.91 |
| 76         | 189.38              | 190.23 |
| 102        | 259.81              | 259    |
