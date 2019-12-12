import torch.nn as nn

class Network(nn.Module):

    def __init__(self, vocab_size, embed_size, label_size):
        super(Network, self).__init__()
        self.vocab_size =  vocab_size
        self.embed_size = embed_size
        self.label_size = label_size
        self.lstm_hidden_size = 64

        self.embed = nn.EmbeddingBag(self.vocab_size, self.embed_size, sparse=True)
        self.lstm = nn.LSTM(self.embed_size, hidden_size=self.lstm_hidden_size, num_layers=2, dropout=0.8, bidirectional=False)
        self.fc = nn.Linear(self.lstm_hidden_size, self.label_size)

        self._init_weights()

    def _init_weights(self, init_range=0.5):

        self.embed.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0.0)

    def forward(self, text, offsets):
        x = self.embed(text, offsets)
        x = x.reshape(-1, 1, self.embed_size)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x.reshape(-1, self.lstm_hidden_size)
        x = self.fc(x)
        return x


