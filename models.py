import torch
from torch import nn, optim

class SingleInputLSTMModel(nn.Module):
    def __init__(self, vocab_size, arch):
        super(SingleInputLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, arch['embedding_size'])
        self.dropout = nn.Dropout(p=arch['dropout_prob'])
        self.lstm1 = nn.LSTM(
            arch['embedding_size'],
            arch['hidden_units_lstm'],
            num_layers=arch['num_layers_lstm'],
            bidirectional=False
        )
        self.dense = nn.Linear(arch['hidden_units_lstm'], vocab_size)
        self.log_softmax = nn.LogSoftmax(-1)

        self.arch = arch

    def init_weights(self):
        initrange = self.arch['init_range']
        self.lstm1.weight.data.uniform_(-initrange, initrange)
        self.dense.bias.data.zero_()
        self.dense.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # tuple of (hidden state, cell state)
        return (
            torch.zeros(self.arch['num_layers_lstm'], batch_size, self.arch['hidden_units_lstm']).float(),
            torch.zeros(self.arch['num_layers_lstm'], batch_size, self.arch['hidden_units_lstm']).float()
        )

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        dropped = self.dropout(embedded)
        output_seq, hidden_output = self.lstm1(dropped, hidden)

        # dense layer, and flatten batches/seq length so we can apply softmax to all output vectors
        dense_output = self.dense(output_seq)
        dense_output = dense_output.view(-1, dense_output.shape[-1])

        # log softmax for smoother training
        normalized_output = self.log_softmax(dense_output)
        
        return normalized_output, hidden_output

class DualInputLSTMModel(nn.Module):
    def __init__(self, vocab_size1, vocab_size2, arch):
        super(DualInputLSTMModel, self).__init__()
        self.embedding1 = nn.Embedding(vocab_size1 + 1, arch['embedding_size_1'])
        self.embedding2 = nn.Embedding(vocab_size2 + 1, arch['embedding_size_2'])
        self.dropout = nn.Dropout(p=arch['dropout_prob'])
        self.lstm1 = nn.LSTM(
            arch['embedding_size_1'] + arch['embedding_size_2'],
            arch['hidden_units_lstm'],
            num_layers=arch['num_layers_lstm'],
            bidirectional=False
        )
        self.dense = nn.Linear(arch['hidden_units_lstm'], vocab_size1)
        self.log_softmax = nn.LogSoftmax(-1)

        self.arch = arch

    def init_weights(self):
        initrange = self.arch['init_range']
        self.lstm1.weight.data.uniform_(-initrange, initrange)
        self.dense.bias.data.zero_()
        self.dense.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        # tuple of (hidden state, cell state)
        return (
            torch.zeros(self.arch['num_layers_lstm'], batch_size, self.arch['hidden_units_lstm']).float(),
            torch.zeros(self.arch['num_layers_lstm'], batch_size, self.arch['hidden_units_lstm']).float()
        )

    def forward(self, input, hidden):
        input_1, input_2 = torch.split(input, 1, -1)
        embedded_1 = self.embedding1(input_1.squeeze(-1))
        embedded_2 = self.embedding2(input_2.squeeze(-1))
        concatenated = torch.cat([embedded_1, embedded_2], -1)
        dropped = self.dropout(concatenated)
        output_seq, hidden_output = self.lstm1(dropped, hidden)

        # dense layer, and flatten batches/seq length so we can apply softmax to all output vectors
        dense_output = self.dense(output_seq)
        dense_output = dense_output.view(-1, dense_output.shape[-1])

        # log softmax for smoother training
        normalized_output = self.log_softmax(dense_output)
        
        return normalized_output, hidden_output
