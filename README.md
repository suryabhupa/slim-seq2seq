## slim_seq2seq
A distillation of the official Tensorflow seq2seq model and tutorial.

## Files

`seq2seq.py`: Contains the implementations of the encoder and decoder for the basic RNN seq2seq model and for the attention-based RNN seq2seq model.

`seq2seq_model.py`: Contains the code for processing minibatches with bucketing and accumulating gradients.

`translate.py`: End-to-end implementation of the NMT problem; defines global variables, loads data, and contains code for running training and decoding.

`data_utils.py`: Utility methods for loading, tokenizing, and preprocessing English-to-French corpus.

## Tutorial

The main logic for starting the training, as well as decoding a particular sentence, lies in `translate.py`. All flags are passed to this file, where the user has a choice between training the model, or decoding a provided sentence (translating it to French) using a trained model. The specifics of this are detailed in the first set of comments in `translate.py` in lines X-Y.

##### `translate.py`

There are four functions in `translate.py`: `read_data()`, `create_model()`, `train()`, `decode()`; the first two are convenience functions to prepare the data and model, the train function contains the end-to-end training code, and the decode function is used for interactive translation. Each method is explained in more detail here:

`read_data`: Reads training data (English and French sentences) for the source language and the target language; used in the main training loop.

`create_model`: Create attention-based seq2seq model based on FLAGS or loads a checkpointed model from a previous run from a specified training directory.

`train`: Runs the computation graph; the rough steps done in the training loop are the following. First, the preparation steps:

`decode`: Loads a saved model and runs a single forward pass of the model through a provided English sentence, outputting the greedily decoded French translation.

###### `train()`

A more elaborate treatment of the `train()` function is presented here, and the high level overview of the function follows:

1. Load the training and development sets for both the English and French pre-processed sentences.
2. Instantiate the TensorFlow session.
3. Create model using `create_model()`
4. Read data using `read_data()` and compute sizes of corresponding buckets.

Now that all data is necessary and loaded (likely onto GPUs, or whatever are the available computing resources), we can begin running the computation graph and updating our model.

1) The data is fetched by `get_batch` and fed into the model via the placeholders:

```
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
```
In this line, the next batch is extracted, and each of the `encoder_inputs`, `decoder_inputs`, and `target_weights` are the placeholders for the input data.

2) The model takes a single step forward in processing this batch, and the loss for this step is returned and later accumulated.

```
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      ...
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1
```
NOTE: Unless the boolean `forward_only` is set to false, the gradients computed in the `step()` method are not used to update the model's parameters. This update procedure is done in `step()`, but is skipped if only forward propagation is specified.

3) Repeat Steps 1 and 2 for n=`FLAGS.steps_per_checkpoint` updates, or for 1 epoch. Note that often times in training deep models, an epoch does not necessarily have to complete one pass through the training data and can instead be completed by a fixed number of updates to the model, as is done here.

4) Checkpoint the model:
  * Compute perplexity for the previous training epoch.
  * Anneal learning rate if there are no improvements over last three epochs.
  * Save the model to `train_dir`
  * Evaluate model thus far on development set and report perplexity.

5) Repeat Steps 1-4.

###### `decode()`

Decode follows the exact same flow as `train()` does. The session is instantiated, and the model is created using `create_model`. The second argument passed is the `forward_only` flag, and in this case, is set to true since we only decode a sentence and freeze the weights. The batch size is set to 1 since we only process one sentence (note that this would be both incredibly inefficient computationally as well as lead to noisy gradient samples and potentially unstable training). The sentence is tokenized, bucketed, and fed to the main computation graph as usual, with the output_logits decded to symbols and finally outputted.

##### `seq2seq_model.py`

##### `seq2seq.py`

##### `data_utils.py`
