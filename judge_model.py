import torch.nn as nn
import torch


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
        self.judge_drop_em = nn.Dropout(dropout_em)
        self.judge_encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.judge_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout_rnn)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.judge_rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout_rnn)
        self.judge1_out = nn.Sequential(
            nn.Linear(nhid, n_cl_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_out),
            nn.Linear(n_cl_hidden, nclass)
        )
        self.judge_out = nn.Sequential(
            nn.Linear(nclass*2, nclass*2),
            nn.ReLU(),
            nn.Linear(nclass*2, 1),
            nn.Sigmoid()
        )

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder_judge.weight = self.judge_encoder.weight

        self.init_weights()
        # initialize the weights for the judge network
        self.judge_out.apply(weights_init)
        self.judge1_out.apply(weights_init)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.judge_encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, last_location, g_label):
        emb = self.judge_drop_em(self.judge_encoder(input))
        output, hidden = self.judge_rnn(emb, hidden)
        decoded = self.judge1_out(output[last_location-2, range(input.size()[1])])
        D_flabel = torch.cat((decoded, g_label), 1)
        judge_out = self.judge_out(D_flabel)
        return judge_out

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
