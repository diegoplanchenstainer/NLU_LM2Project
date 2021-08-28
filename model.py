import torch.nn as nn
from my_lstm import MyLSTM


class RNNLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, nhid, nlayers, dropout=0.5, residual_con=False, internal_drop=False, init_f_bias=False):
        super(RNNLM, self).__init__()
        self.ntoken = ntoken
        self.nhid = nhid
        self.nlayers = nlayers
        self.residual_con = residual_con
        self.internal_drop = internal_drop
        self.init_f_bias = init_f_bias

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(self.ntoken, self.nhid)

        self.rnns = nn.ModuleList()
        for _ in range(self.nlayers):
            self.rnns.append(MyLSTM(self.nhid, self.residual_con, self.internal_drop, self.init_f_bias))
        self.decoder = nn.Linear(self.nhid, self.ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        output = self.encoder(input)
        for i in range(self.nlayers):
            output = self.drop(output)
            output, hidden[i] = self.rnns[i](output, hidden[i])

        decoded = self.decoder(self.drop(output))
        decoded = decoded.view(-1, self.ntoken)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        return (weight.new_zeros(bsz, self.nhid),
                weight.new_zeros(bsz, self.nhid))
