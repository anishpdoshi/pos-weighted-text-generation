import pdb
import pprint
import tqdm

import nltk
import torch
from torch import nn, optim

from models import SingleInputLSTMModel, DualInputLSTMModel
from data import reduced_text, to_sentence
from featurization import create_dictionaries, create_pos_dictionaries, bow
from decode import constrained_beam_search

def batchify(tensor, batch_size, seq_length, seq_first=True):
    block_size = batch_size * seq_length
    limit = (len(tensor) // block_size) * block_size
    trimmed = tensor.narrow(0, 0, limit)
    batched = trimmed.view(limit // block_size, batch_size, seq_length, -1).squeeze()
    if seq_first:
        batched = torch.transpose(batched, 1, 2)
    return batched

def train(model, X, y, hp, on_tenth_epoch= lambda: None, save_path=None):
    loss_fn = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters())
    
    hidden = model.init_hidden(hp['batch_size'])
    try:
        for epoch in range(hp['num_epochs']):
            model.train()
            total_loss = 0
            shuffle_ind = torch.randperm(len(X))
            shuffled_X, shuffled_y = X[shuffle_ind], y[shuffle_ind]
            split_index = int(hp['train_split'] * len(X))

            train_X, train_y = shuffled_X[:split_index], shuffled_y[:split_index]
            validation_X, validation_y = shuffled_X[split_index:], shuffled_y[split_index:]
            
            # train_batches, validation_batches = shuffled[:split_index], shuffled[split_index:]

            for inputs, labels in tqdm.tqdm(zip(train_X, train_y)):
                optimizer.zero_grad()
                hidden = (hidden[0].detach(), hidden[1].detach())
                loss = 0
                # Teacher forcing - add losses per timestep
                if torch.rand(1).item() <= hp['teacher_forcing_prob']:
                    for i in range(0, hp['seq_length'] - 2):
                        t_inputs, t_labels = inputs[i].unsqueeze(0), inputs[i + 1]
                        output, hidden = model(t_inputs, hidden)
                        loss += nn.NLLLoss()(output, t_labels.flatten())
                else:
                    output, hidden = model(inputs, hidden)

                    loss = loss_fn(output, labels.flatten())

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()

                total_loss += loss.item()

            model.eval()
            validation_loss = 0
            for val_inputs, val_labels in zip(validation_X, validation_y):
                val_hidden = (hidden[0].detach(), hidden[1].detach())
                # val_inputs, val_labels = batch.T[:-1], batch.T[1:]
                val_output, val_h = model(val_inputs, val_hidden)
                validation_loss += loss_fn(val_output, val_labels.flatten()).item()

            if epoch % 10 == 0:
                on_tenth_epoch()

            print('Epoch {} Loss: {}, ValLoss: {}'.format(epoch + 1, total_loss, validation_loss))
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_path)
    except (KeyboardInterrupt, SystemExit):
        if save_path is not None:
            print('interrupted, saving')
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, save_path)


if __name__ == '__main__':
    reduced = reduced_text('lfrankbaum', 'wonderful-wizard-of-oz')
    words, pos_tags = zip(*reduced)
    word_model_hyperparams = {
        'batch_size': 32,
        'seq_length': 20,
        'num_epochs': 1000,
        'teacher_forcing_prob': 0.0,
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

    dual_model_hyperparams = {
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

    word_info = create_dictionaries(words)
    pos_info = create_dictionaries(pos_tags)

    word_model = SingleInputLSTMModel(word_info['vocab_size'], word_model_hyperparams['arch'])
    dual_model = DualInputLSTMModel(pos_info['vocab_size'], word_info['vocab_size'], dual_model_hyperparams['arch'])

    # TODO - runner script
    state = 'eval'

    if state == 'train':

        word_data = torch.tensor(word_info['to_indices'](words))
        pos_data = torch.tensor(pos_info['to_indices'](pos_tags))

        word_batches = batchify(word_data, word_model_hyperparams['batch_size'], word_model_hyperparams['seq_length'])
        pos_batches = batchify(pos_data, dual_model_hyperparams['batch_size'], dual_model_hyperparams['seq_length'])
        dual_batches = torch.stack([pos_batches, word_batches], -1)

        word_X, word_y = word_batches[:,:-1], word_batches[:,1:]
        dual_X, dual_y = dual_batches[:,:-1], pos_batches[:,1:]

        train(word_model, word_X, word_y, word_model_hyperparams, save_path='word_model.pt')
        train(dual_model, dual_X, dual_y, dual_model_hyperparams, save_path='dual_word_pos.pt')
    else:

        word_model.load_state_dict(torch.load('word_model.pt')['state_dict'])
        dual_model.load_state_dict(torch.load('dual_model.pt'))

        constraint_map = bow(create_pos_dictionaries(reduced), word_info['w_to_i'], pos_info['w_to_i'])
        def get_constraint_vector(sequence):
            word_seq = [word_info['i_to_w'][i] for i in sequence]
            pos_seq = [pos_info['w_to_i'][tag] for w, tag in nltk.pos_tag(word_seq)]
            stacked_seq = torch.stack([torch.tensor(pos_seq), torch.tensor(sequence)], -1)

            dual_init_hidden = dual_model.init_hidden(1)

            tag_output, _ = dual_model(stacked_seq.unsqueeze(1), dual_init_hidden)
            next_tag = torch.argmax(tag_output[-1])

            return constraint_map[next_tag.item()]
            # return torch.ones(word_info['vocab_size'])

        word_model.eval()
        dual_model.eval()

        initial_sentence = 'kansas'.split(' ')
        initial_indices = [word_info['w_to_i'][word] for word in initial_sentence]
        test_output, test_hidden = word_model(torch.tensor(initial_indices).unsqueeze(0).T, word_model.init_hidden(1))

        results = constrained_beam_search(
            word_model,
            test_output[-1],
            test_hidden,
            get_constraint_vector,
            100,
            beam_width = 10
        )

    play_seq = torch.stack([torch.ones(3), torch.zeros(3)], -1)
    dual_init_hidden = dual_model.init_hidden(1)
    tag_output = dual_model(play_seq.unsqueeze(1).long(), dual_init_hidden)

    pprint.pprint([
        (' '.join([word_info['i_to_w'][i] for i in seq]), prob)
        for seq, _, prob in results[:5]
    ])
    # def epoch_hook():
    #     model.eval()

    #     initial_sentence = 'but rather in the lap of being ,'.split(' ')
    #     initial_indices = [w_to_i[word] for word in initial_sentence]
    #     test_output, test_hidden = model(torch.tensor(initial_indices).unsqueeze(0).T, model.init_hidden(1))

    #     results = beam_search(model, test_output[-1], test_hidden, vocab_size, 50, beam_width = hp['beam_width'])
    #     pprint.pprint([(to_sentence([i_to_w[i] for i in seq]), prob) for seq, _, prob in results])

    

