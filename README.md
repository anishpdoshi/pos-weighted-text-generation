# Part-of-Speech weighted text generation

## Introduction

Over the last few decades, the task of _text generation_ has risen in both effectiveness and applicability. Given a sequence of word tokens, text generator models aim predict the most likely next word in the sequence, which can be done repeatedly to generate entire sentences and paragraphs. 

This project augments a _relatively_ lightweight text generation model (a tri-layer, forward LSTM with 512 hidden units per layer) with a _part-of-speech model_. The POS model works as follows - given an input sequence of both word tokens _and_ their associated part-of-speech tags (generated from `nltk`'s existing POS tagger), the model attempts to predict the most likely next pos tag (just the pos tag - not the next word!) This model (another stacked LSTM model) is trained independently (for now) from the text generation model.

To generate words, we use beam search to decode the trained text generation model. Before computing the topk probabilities for each beam's next word, however, we first run the POS model on each beam's current sequence to compute a distribution over the possible next _pos tag_ for that beam. This distribution over pos tags is then used to compute a distribution over _words_, by using a tensor that associates pos tags to words based on the input corpus words' tags. The pos computed distribution over words is multiplied by the original text generation distribution, and only then do we find the topk probabilities across beams. 

The inclusion of this auxiliary model allows this model to generate text comparable to deeper networks, without the training drawbacks of larger networks (e.g., stacked transformers).

## Usage

**1. Prerequisites**
This project relies on PyTorch, nltk (with punkt and perceptron taggers), and tqdm:

  pip install torch==1.5.0 nltk==3.4.5 tqdm==4.45.0

  >> nltk.download('punkt')
  >> nltk.download('averaged_perceptron_tagger')

**2. Training**

You can then run `sequence_prediction.py` to train the text generation and POS models on given data. Included are models pretrained on a number of works from L. Frank Baum.

**3. Generation**
Trained models can then be used to generate sentences.



