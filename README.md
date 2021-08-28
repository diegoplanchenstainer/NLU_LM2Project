# NLU_LM2 Final Project
## LM2 Project for NLU course a.a. 2020/2021

LM2: Implement an RNN Language Model using one of the most famous architectures i.e. Vanilla (aka Elman neural network), LSTM or GRU.

Student: Diego Planchenstainer (223728)

## Requirements
`Python` and the `pytorch` package are needed to run this code.

Guide to install pytorch: https://pytorch.org/get-started/locally/

Dataset: Penn Treebank, you can find it here: https://deepai.org/dataset/penn-treebank
The dataset should be contained in a folder called `dataset` that will contain `ptb.train.txt`, `ptb.valid.txt` and `ptb.test.txt`

The backbone of this code can be found at: https://github.com/pytorch/examples/blob/master/word_language_model

This model has 3 configuration variables in which it can be set the following:
- internal dropout
- residual connection
- initialization of forget bias

The best result was obtained with 2-layer with residual connection