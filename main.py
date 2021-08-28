import time
import math
import torch
import os
import torch.nn as nn
import torch.onnx
from torch.nn.utils.rnn import pad_sequence

from pennTreebank import Corpus
import model

from tensorboardX import SummaryWriter
import pandas as pd


path = "./dataset/"
nhid = 600
nlayers = 2
clip = 0.25
dropout = 0.5

residual_con = True
internal_drop = False
init_f_bias = True

epochs = 30
lr = 1e-3  # 20
batch_size = 64
eval_batch_size = 1

log_interval = 50
save = "model.pt"

skip_training = False
load_model = False

seed = 1113
torch.manual_seed(seed)

name = "runs/exp" + str(len(os.listdir("runs"))+1)
os.makedirs(name)

df = pd.DataFrame(columns=["epoch", "train_loss",
                           "train_ppl", "val_loss", "val_ppl"])

writer = SummaryWriter(log_dir=name)


def log_values(writer, epoch, loss, ppl, prefix):
    writer.add_scalar(f"{prefix}/loss", loss, epoch)
    writer.add_scalar(f"{prefix}/ppl", ppl, epoch)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Load data
###############################################################################

print("Loading data...")
corpus = Corpus(path)

train_data = corpus.train
val_data = corpus.valid
test_data = corpus.test


###############################################################################
# Build the model
###############################################################################

print("Building model...")

ntokens = len(corpus.dictionary)

model = model.RNNLM(ntokens,
                    nhid, nlayers, dropout, residual_con, internal_drop, init_f_bias).to(device)
if load_model:
    with open('model.pt', 'rb') as f:
        model = torch.load(f)
        print('Model loaded')

# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss(
    ignore_index=corpus.dictionary.word2idx['<pad>'])
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1.2e-6)


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, batch_size):
    # get a batch of equally long sentences (adds <pad> to fill at the same lenght)
    padded_data = []
    padded_target = []
    index = i*batch_size

    padded_data = [data[:-1] for data in source[index:index+batch_size]]
    padded_target = [data[1:] for data in source[index:index+batch_size]]

    padded_data = pad_sequence(
        padded_data, padding_value=corpus.dictionary.word2idx['<pad>'], batch_first=True)
    padded_target = pad_sequence(
        padded_target, padding_value=corpus.dictionary.word2idx['<pad>'], batch_first=True)
    return padded_data.transpose(0, 1).contiguous(), padded_target.transpose(0, 1).contiguous()


def evaluate(data_source):

    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for i in range(0, len(data_source)//eval_batch_size):
            hidden = [model.init_hidden(eval_batch_size)
                      for _ in range(nlayers)]
            data, targets = get_batch(data_source, i, eval_batch_size)
            output, hidden = model(data, hidden)
            hidden = [repackage_hidden(h) for h in hidden]
            total_loss += data.size(1) * criterion(output,
                                                   targets.view(-1)).item()
    return total_loss / (len(data_source))


def train():

    model.train()
    total_loss = 0.
    gloabal_loss = 0.
    start_time = time.time()

    hidden = [model.init_hidden(batch_size) for _ in range(nlayers)]
    for i in range(0, len(train_data)//batch_size):
        data, targets = get_batch(train_data, i, batch_size)

        optimizer.zero_grad()

        hidden = [repackage_hidden(h) for h in hidden]
        output, hidden = model(data, hidden)
        loss = criterion(output, targets.view(-1))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()
        gloabal_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.0f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, i, len(train_data) // batch_size, lr,
                      elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    return gloabal_loss / (len(train_data)//batch_size)


def generate_txt(nwords):
    model.eval()
    with torch.no_grad():
        with open('sample.txt', 'w') as f:

            # Set intial hidden and cell states
            hidden = [(torch.zeros(1, nhid).to(device),
                       torch.zeros(1, nhid).to(device)) for _ in range(nlayers)]

            input = torch.tensor(
                [[corpus.dictionary.word2idx['<sos>']]], device=device)
            for i in range(nwords):
                # Forward propagate RNN
                output, hidden = model(input, hidden)

                # Sample a word id
                prob = output.exp()[-1:]
                word_id = torch.multinomial(prob, num_samples=1).item()

                tensor_word = torch.tensor([[word_id]], device=input.device)
                input = torch.cat((input, tensor_word), 0)
                # File write
                word = corpus.dictionary.idx2word[word_id]
                if word == '<eos>':
                    word = '\n'
                    input = torch.tensor(
                        [[corpus.dictionary.word2idx['<sos>']]], device=device)
                else:
                    word += ' '
                f.write(word)

                if (i+1) % 100 == 0:
                    print(
                        'Sampled [{}/{}] words and save to {}'.format(i+1, nwords, 'sample.txt'))


# Loop over epochs.
lr = lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
if not skip_training:
    try:
        print("Init training...")
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            train_loss = train()

            log_values(writer, epoch, train_loss,
                       math.exp(train_loss), "Training")

            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)

            log_values(writer, epoch, val_loss,
                       math.exp(val_loss), "Validation")

            df.loc[len(df)+1] = [epoch, train_loss,
                                 math.exp(train_loss), val_loss,  math.exp(val_loss)]

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

writer.close()
print("Writing excel file...")
with pd.ExcelWriter(name + "/logs.xlsx") as excell_wrt:
    df.to_excel(excell_wrt)

# Load the best saved model.
with open(save, 'rb') as f:
    model = torch.load(f)
    print('Model loaded')

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

generate_txt(250)
