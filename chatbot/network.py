import torch
import torch.nn as nn
import torch.nn.functional as F

from chatbot.config import SOS_token


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.n_layers,
                          dropout=(0 if self.n_layers==1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        #input_seq shape: [max_seq_len, batch_size]
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        # after GRU,
        # outputs shape [max_seq_len, batch_size, 2*hidden_size]
        # hidden shape [2, batch_size, hidden_size]

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # sum bidirectional GRU outputs,
        # final outputs shape [max_seq_len, batch_size, hidden_size]
        outputs= outputs[:, :, 0:self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden

class Attn(nn.Module):
    '''
    Luong attention layer "Global attention"
    '''
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(2*self.hidden_size, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output))).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        else:
            attn_energies = self.dot_score(hidden, encoder_outputs)

        #the shape of attn_energies is now [max_seq_len, batch_size]

        # transpose attn_energies,
        # now the shape will be [batch_size, max_seq_len]
        attn_energies = attn_energies.t()

        # return the softmax normalized probability score (with added dimension),
        # the final shape of attention weights will be [batch_size, 1, max_seq_len]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_method, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        #layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.n_layers,
                          dropout=(0 if self.n_layers==1 else dropout),
                          bidirectional=False)
        self.attn = Attn(self.attn_method, hidden_size)
        self.concat = nn.Linear(2*self.hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        #get embedding of the input_step(current word), input_step shape: [1, batch_size]
        #after embedding, the embedded shape: [1, batch_size, hidden_size]
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        #forward through undirectional GRU
        #rnn_output shape: [1, batch_size, hidden_size]
        #hidden shape: [n_layers, batch_size, hidden_size]
        rnn_output, hidden = self.gru(embedded, last_hidden)

        #calculate attention weights from current GRU output
        #weights shape: [batch_size, 1, max_seq_len]
        attn_weights = self.attn(rnn_output, encoder_outputs)

        #multiply attention weights to encoder outputs to get new "weighted sum" context vector
        #encoder_outputs shape [max_seq_len, batch_size, hidden_size]
        #weights shape: [batch_size, 1, max_seq_len]
        #the context shape: [batch_size, 1, hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        #concatenate weighted context vector and GRU output using Luong eq.5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        #predict next wordusing Luong eq. 6
        #output shape [batch_size, output_size]
        output = self.out(concat_output)
        output = torch.nn.functional.softmax(output, dim=1)

        return output, hidden

class GreedySearchDecoder(nn.Module):
    def __init__(self, device, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        #forward the batch of input_seq through the encoder
        encoder_output, encoder_hidden = self.encoder(input_seq, input_length)

        #prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        #initialize decoder input with SOS_token
        decoder_input = torch.LongTensor([[SOS_token]], device=self.device)

        #initialize tensors to append decoded words
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)

            #obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            #record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            #prepare current token to be next (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, dim=0)

        return all_tokens, all_scores










