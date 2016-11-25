## slim_seq2seq
A distillation of the official Tensorflow seq2seq model and tutorial.

## Files

`seq2seq.py`: Contains the implementations of the encoder and decoder for the basic RNN seq2seq model and for the attention-based RNN seq2seq model.

`seq2seq_model.py`: Contains the code for processing minibatches with bucketing and accumulating gradients.

`translate.py`: End-to-end implementation of the NMT problem; defines global variables, loads data, and contains code for running training and decoding.

`data_utils.py`: Utility methods for loading, tokenizing, and preprocessing English-to-French corpus.

## Tutorial
