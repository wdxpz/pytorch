import torch.nn as nn

class Network(nn.Module):

    def __init__(self, vocab_size, embed_size, label_size):
        super(Network, self).__init__()
        self.vocab_size =  vocab_size
        self.embed_size = embed_size
        self.label_size = label_size

        self.embed = nn.EmbeddingBag(self.vocab_size, self.embed_size, sparse=True)
        self.fc = nn.Linear(self.embed_size, self.label_size)

        self._init_weights()

    def _init_weights(self, init_range=0.5):

        self.embed.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0.0)

    def forward(self, text, offsets):
        x = self.fc(self.embed(text, offsets))
        return x


