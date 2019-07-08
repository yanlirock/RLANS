import torch.nn as nn


def weights_init(m):
    initrange = 0.1
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-initrange, initrange)
        m.bias.data.zero_()

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, nclass, 
                 dropout_em=0.5,dropout_rnn=0,dropout_out=0, tie_weights=False, n_cl_hidden=30):
        super(RNNModel, self).__init__()
        self.drop_em = nn.Dropout(dropout_em)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout_rnn)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout_rnn)
        self.dis_out = nn.Sequential(
            nn.Linear(nhid, n_cl_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_out),
            nn.Linear(n_cl_hidden, nclass)
        )

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder_dis.weight = self.encoder.weight

        self.init_weights()
        self.dis_out.apply(weights_init)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, last_location):
        emb = self.drop_em(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        decoded = self.dis_out(output[last_location-2, range(input.size()[1])])
        return decoded

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
