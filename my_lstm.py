import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLSTM(nn.Module):
    def __init__(self, nhid, residual_con=False, internal_drop=False, init_f_bias=False):
        super(MyLSTM, self).__init__()
        self.nhid = nhid
        self.residual_con = residual_con
        self.internal_drop = internal_drop
        self.init_f_bias = init_f_bias

        # Forget gate
        self.forget_gate = nn.Linear(nhid * 2, nhid, bias=True)

        # Input gate
        self.it_gate = nn.Linear(nhid * 2, nhid, bias=True)

        # cell_state
        self.gt_gate = nn.Linear(nhid * 2, nhid, bias=True)

        # o_t
        self.ot_gate = nn.Linear(nhid * 2, nhid, bias=True)

        if internal_drop:
            self.internal_dropout = nn.Dropout()

        if self.init_f_bias:
            nn.init.constant_(self.forget_gate.bias, 1)

    def forward(self, x, init_states=None):

        seq_size, _, _ = x.size()
        hidden_seq = []

        hidden_state, cell_state = init_states

        for t in range(seq_size):
            x_t = x[t, :, :]

            x_con_h = torch.cat((x_t, hidden_state), dim=1)

            # forget gate
            ft = torch.sigmoid(self.forget_gate(x_con_h))

            # input gate
            it = torch.sigmoid(self.it_gate(x_con_h))
            gt = torch.tanh(self.gt_gate(x_con_h))

            ot = torch.sigmoid(self.ot_gate(x_con_h))

            #  Semeniuta et alia approach
            if self.internal_drop:
                mul = self.internal_dropout(it*gt)
            else:
                mul = it*gt

            cell_state = ft * cell_state + mul

            hidden_state = ot * torch.tanh(cell_state)

            hidden_seq.append(hidden_state.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        if self.residual_con:
            return hidden_seq, (hidden_state, cell_state)
        else:
            return x + hidden_seq, (hidden_state, cell_state)
