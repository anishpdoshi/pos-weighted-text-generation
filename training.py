import os
import pdb
import pprint
import tqdm

import torch
from torch import nn, optim

def batchify(tensor, batch_size, seq_length, seq_first=True):
    block_size = batch_size * seq_length
    limit = (len(tensor) // block_size) * block_size
    trimmed = tensor.narrow(0, 0, limit)
    batched = trimmed.view(limit // block_size, batch_size, seq_length, -1).squeeze()
    if seq_first:
        batched = torch.transpose(batched, 1, 2)
    return batched

def save(model, optimizer, path):
    data = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(data, path)

def initialize(model, optimizer, path):
    data = torch.load(path)
    model.load_state_dict(data['model_state'])
    optimizer.load_state_dict(data['optimizer_state'])

#  TODO - support specifying regular epoch operations, e.g. on_tenth_epoch function argument
def train(model, X, y, hp, path):
    loss_fn = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters())

    if os.path.exists(path):
        print('Path {} already exists - picking up from there'.format(path))
        initialize(model, optimizer, path)
    
    hidden = model.init_hidden(hp['batch_size'])
    try:
        for epoch in range(hp['num_epochs']):
            model.train()
            total_loss = 0
            shuffle_ind = torch.randperm(len(X))
            shuffled_X, shuffled_y = X[shuffle_ind][:100], y[shuffle_ind][:100]
            split_index = int(hp['train_split'] * len(shuffled_X))

            train_X, train_y = shuffled_X[:split_index], shuffled_y[:split_index]
            validation_X, validation_y = shuffled_X[split_index:], shuffled_y[split_index:]

            for inputs, labels in tqdm.tqdm(zip(train_X, train_y), total=len(train_X)):
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
                val_output, val_h = model(val_inputs, val_hidden)
                validation_loss += loss_fn(val_output, val_labels.flatten()).item()

            print('Epoch {} Loss: {}, ValLoss: {}'.format(epoch + 1, total_loss, validation_loss))
        if path is not None:
            save(model, optimizer, path)
    except (KeyboardInterrupt, SystemExit):
        if path is not None:
            print('signal interrupted, saving model')
            save(model, optimizer, path)

