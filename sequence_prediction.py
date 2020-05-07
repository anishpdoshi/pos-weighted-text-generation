import os
import pprint
import argparse

import torch

from models import SingleInputLSTMModel, DualInputLSTMModel
from data import reduced_text, to_sentence
from featurization import Corpus
from training import train, batchify
from decoding import generate_sentences


parser = argparse.ArgumentParser(description='Using POS tagging to help decode a language model')

# todo - support hyperparams and architectures
parser.add_argument('--path', help='filepath of textfile/directory to load data from. if path is a directory, all files in the directory will be parsed.')
parser.add_argument('--wordpath', help='path for an existing word model file')
parser.add_argument('--pospath', help='path for an existing pos model file')
parser.add_argument('--pos-constraint', default='weighted', choices=['weighted', 'strict', 'none'], help='type of POS constraining to use in decoding')
parser.add_argument('--sentence', default=', and', help='initial sentence or phrase with which we will seed our example language generation')
parser.add_argument('action', choices=['train', 'train-word', 'train-pos', 'gen'], help='whether to train (all, word model, pos modoel) or just load everything for evaluation')
args = parser.parse_args()

if __name__ == '__main__':
    corpus = Corpus(reduced_text(args.path))
    print('Initialized corpus - {} items, {} unique words, {} unique tags'.format(corpus.text_length, corpus.num_words, corpus.num_tags))

    word_model_hyperparams = {
        'batch_size': 32,
        'seq_length': 20,
        'num_epochs': 1000,
        'teacher_forcing_prob': 0.5,
        'beam_width': 5,
        'train_split': 0.85,
        'arch': {
            'init_range': 0.2,
            'dropout_prob': 0.5,
            'embedding_size': 50,
            'hidden_units_lstm': 512,
            'num_layers_lstm': 3,
            'hidden_units_dense': 512,
        }
    }

    pos_model_hyperparams = {
        'batch_size': 32,
        'seq_length': 20,
        'num_epochs': 200,
        'teacher_forcing_prob': 0.0,
        'beam_width': 5,
        'train_split': 0.85,
        'arch': {
            'init_range': 0.2,
            'dropout_prob': 0.5,
            'embedding_size_1': 10,
            'embedding_size_2': 50,
            'hidden_units_lstm': 256,
            'num_layers_lstm': 3,
            'hidden_units_dense': 256,
        }
    }


    word_model = SingleInputLSTMModel(corpus.num_words, word_model_hyperparams['arch'])
    pos_model = DualInputLSTMModel(corpus.num_tags, corpus.num_words, pos_model_hyperparams['arch'])

    wordpath = args.wordpath or args.author + '_word_model.pt'
    pospath = args.pospath or args.author + '_pos_model.pt'

    if os.path.exists(wordpath):
        word_model.load_state_dict(torch.load('word_model.pt')['model_state'])
    if os.path.exists(pospath):
        pos_model.load_state_dict(torch.load('pos_model.pt'))['model_state']

    if args.action != 'gen':
        word_batches = batchify(corpus.words_data, word_model_hyperparams['batch_size'], word_model_hyperparams['seq_length'])
        pos_batches = batchify(corpus.tags_data, pos_model_hyperparams['batch_size'], pos_model_hyperparams['seq_length'])
        combined_batches = torch.stack([pos_batches, word_batches], -1)

        word_X, word_y = word_batches[:,:-1], word_batches[:,1:]
        pos_X, pos_y = combined_batches[:,:-1], pos_batches[:,1:]

        if args.action in ['train', 'train-word']:
            train(word_model, word_X, word_y, word_model_hyperparams, wordpath)
        if args.action in ['train', 'train-pos']:
            train(pos_model, pos_X, pos_y, pos_model_hyperparams, pospath)

    generated = generate_sentences(args.sentence, word_model, pos_model, corpus, constraint_type='weighted')
    pprint.pprint(generated[:5])
    

